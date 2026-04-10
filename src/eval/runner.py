"""
Evaluation runner for Memory-R1 experiments.

Orchestrates model loading, inference, judging, and metrics across all
configs and benchmarks defined in the eval YAML config.

Supports two inference backends:
  - "hf"   : HuggingFace Transformers model.generate() with manual batching
  - "vllm" : vLLM offline LLM engine with continuous batching + PagedAttention

Usage:
    python -m src.eval.runner --config configs/eval.yaml
    python -m src.eval.runner --config configs/eval.yaml --backend vllm
    python -m src.eval.runner --config configs/eval.yaml --backend vllm --gpus 4
    python -m src.eval.runner --config configs/eval.yaml --skip-judge
    python -m src.eval.runner --config configs/eval.yaml --judge-only
    python -m src.eval.runner --config configs/eval.yaml --aggregate-only
    python -m src.eval.runner --config configs/eval.yaml --models config_a_full
"""
import argparse
import gc
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from src.common.config import load_config
from src.eval.model_loader import (
    load_model_and_tokenizer,
    load_mm_model,
    load_model_vllm,
    load_mm_model_vllm,
)
from src.eval.inference import (
    format_aa_prompt,
    extract_answer,
    generate_answer,
    generate_answers_batched,
    generate_answers_vllm,
    run_mm_on_sessions,
    run_mm_all_conversations_vllm,
)
from src.eval.metrics import evaluate_predictions, format_results_table, save_results
from src.eval.judge import judge_batch
from src.memory.heuristic import build_heuristic_memories, retrieve_memories

logger = logging.getLogger(__name__)


def load_test_data(test_file):
    """Load test JSONL file."""
    data = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def run_inference(model, tokenizer, model_config, benchmark_config,
                  retrieval_top_k=20, max_examples=None, use_batched=True,
                  inference_batch_size=8, max_new_tokens=1024,
                  mm_model=None, mm_tokenizer=None,
                  backend="hf",
                  output_path=None,
                  prebuilt_memories=None):
    """
    Run inference for a loaded model on a benchmark.
    Groups questions by conversation_id.

    backend="hf": model/tokenizer are HF objects, mm_model/mm_tokenizer are HF objects.
    backend="vllm": model is vllm.LLM, tokenizer is None, mm_model is vllm.LLM, mm_tokenizer is None.

    If output_path is provided, predictions are written incrementally (one JSONL line per prediction).
    If prebuilt_memories is provided (dict of conv_id -> list[str]), uses those instead of running MM.
    """
    model_name = model_config["name"]
    benchmark_name = benchmark_config["name"]
    test_file = benchmark_config["test_file"]
    use_mm = model_config.get("use_mm", False) and mm_model is not None

    logger.info(f"Running {model_name} on {benchmark_name} ({test_file}) [backend={backend}]")
    logger.info(f"Memory construction: {'trained MM' if use_mm else 'heuristic'}")
    test_data = load_test_data(test_file)
    if max_examples and len(test_data) > max_examples:
        test_data = test_data[:max_examples]
        logger.info(f"Capped to {max_examples} examples")
    logger.info(f"Loaded {len(test_data)} test examples")

    conv_groups = defaultdict(list)
    for ex in test_data:
        conv_groups[ex["conversation_id"]].append(ex)
    logger.info(f"Grouped into {len(conv_groups)} unique conversations")

    conv_memories = {}
    all_prompts = []
    all_metadata = []

    # Use prebuilt memories if provided (from sequential vLLM MM phase)
    if prebuilt_memories is not None:
        conv_memories = prebuilt_memories
        logger.info(f"Using {len(conv_memories)} prebuilt conversation memories")
    # Batch MM across all conversations when using vLLM
    elif use_mm and backend == "vllm":
        conv_sessions_map = {}
        for conv_id, examples in conv_groups.items():
            sessions = examples[0].get("sessions", [])
            conv_sessions_map[conv_id] = sessions
        logger.info(f"Building memories via batched MM for {len(conv_sessions_map)} conversations...")
        conv_memories = run_mm_all_conversations_vllm(mm_model, conv_sessions_map)

    for conv_idx, (conv_id, examples) in enumerate(conv_groups.items()):
        if conv_id not in conv_memories:
            sessions = examples[0].get("sessions", [])
            if use_mm:
                conv_memories[conv_id] = run_mm_on_sessions(
                    mm_model, mm_tokenizer, sessions,
                )
            else:
                conv_memories[conv_id] = build_heuristic_memories(sessions)
        memories = conv_memories[conv_id]
        logger.info(
            f"  Conv {conv_idx+1}/{len(conv_groups)} ({conv_id}): "
            f"{len(memories)} memories, {len(examples)} questions"
        )
        for ex in examples:
            question = ex["question"]
            retrieved = retrieve_memories(question, memories, top_k=retrieval_top_k)
            prompt = format_aa_prompt(question, retrieved)
            all_prompts.append(prompt)
            all_metadata.append({
                "question": question,
                "gold_answer": ex["answer"],
                "question_type": ex.get("question_type", "unknown"),
                "source_benchmark": ex.get("source_benchmark", benchmark_name),
                "model": model_name,
                "conversation_id": conv_id,
                "memory_source": "trained_mm" if use_mm else "heuristic",
            })

    # Generate answers
    if backend == "vllm":
        logger.info(f"Generating answers via vLLM ({len(all_prompts)} prompts)...")
        answers = generate_answers_vllm(
            model, all_prompts, max_new_tokens=max_new_tokens,
        )
    elif use_batched and len(all_prompts) > 1:
        logger.info(f"Generating answers (HF batched, batch_size={inference_batch_size})...")
        answers = generate_answers_batched(
            model, tokenizer, all_prompts,
            max_new_tokens=max_new_tokens,
            batch_size=inference_batch_size,
        )
    else:
        logger.info("Generating answers (HF sequential)...")
        answers = []
        for i, prompt in enumerate(all_prompts):
            answer = generate_answer(model, tokenizer,
                                     all_metadata[i]["question"],
                                     [],  # memories already in prompt
                                     max_new_tokens=max_new_tokens)
            answers.append(answer)
            if (i + 1) % 50 == 0:
                logger.info(f"    Progress: {i+1}/{len(all_prompts)}")

    # Combine and write incrementally if output_path provided
    predictions = []
    f_out = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        f_out = open(output_path, "a", encoding="utf-8")
    for meta, answer in zip(all_metadata, answers):
        meta["answer"] = answer
        predictions.append(meta)
        if f_out:
            f_out.write(json.dumps(meta) + "\n")
            f_out.flush()
    if f_out:
        f_out.close()

    logger.info(f"Completed {len(predictions)} predictions")
    return predictions


def print_comparison_table(all_results):
    """Print a cross-model F1 comparison table."""
    print("\n" + "=" * 80)
    print("SUMMARY: Cross-Model Comparison (F1)")
    print("=" * 80)
    benchmarks = sorted({b for r in all_results.values() for b in r})
    header = f"{'Model':<30}" + "".join(f"{b:>15}" for b in benchmarks)
    print(header)
    print("-" * len(header))
    for model_name, model_results in all_results.items():
        row = f"{model_name:<30}"
        for bench in benchmarks:
            if bench in model_results:
                row += f"{model_results[bench]['overall']['f1']:>15.3f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-only", action="store_true",
                        help="Run judge on existing prediction files (no inference)")
    parser.add_argument("--models", type=str, nargs="*")
    parser.add_argument("--benchmarks", type=str, nargs="*")
    parser.add_argument("--retrieval-top-k", type=int, default=None,
                        help="Override retrieval top-k from config")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batched inference (HF backend only)")
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"], default=None,
                        help="Inference backend: 'hf' (HuggingFace) or 'vllm'. Overrides config.")
    parser.add_argument("--gpus", type=int, default=None,
                        help="Number of GPUs for tensor parallelism (vLLM). Overrides config.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Read existing prediction files, recompute metrics, and write all_results.json (no inference/judging)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    config = load_config(args.config)
    hw_cfg = config["evaluation"].get("hardware", {})
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backend: CLI > config > default "hf"
    backend = args.backend or hw_cfg.get("backend", "hf")
    if backend == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError:
            logger.error("vLLM not installed. Install with: pip install 'rl-memory-curriculum[vllm]'")
            sys.exit(1)

    # Hardware settings
    tensor_parallel_size = args.gpus or hw_cfg.get("gpus", 1)
    gpu_memory_utilization = hw_cfg.get("gpu_memory_utilization", 0.85)
    max_model_len = hw_cfg.get("max_model_len", 4096)

    # Retrieval top-k: CLI > config > default 20
    retrieval_top_k = (args.retrieval_top_k
                       or config["evaluation"].get("retrieval", {}).get("top_k", 20))
    inference_batch_size = hw_cfg.get("batch_size", 8)
    max_new_tokens = 1024  # Phase 2 default

    logger.info(f"Backend: {backend}, GPUs (TP): {tensor_parallel_size}")

    # --judge-only: read existing predictions, run judge, recompute metrics
    if args.judge_only:
        judge_cfg = config["evaluation"].get("llm_judge", {})
        model_id = judge_cfg.get("model", "gpt-4o-mini")
        all_results = {}
        for model_cfg in config["evaluation"]["models"]:
            model_name = model_cfg["name"]
            if args.models and model_name not in args.models:
                continue
            model_results = {}
            for bench_cfg in config["evaluation"]["benchmarks"]:
                bench_name = bench_cfg["name"]
                if args.benchmarks and bench_name not in args.benchmarks:
                    continue
                pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
                if not pred_path.exists():
                    logger.warning(f"No predictions found: {pred_path}, skipping")
                    continue
                predictions = []
                with open(pred_path) as f:
                    for line in f:
                        if line.strip():
                            predictions.append(json.loads(line))
                logger.info(f"Judging {len(predictions)} predictions for {model_name}/{bench_name}")
                predictions = judge_batch(predictions, model=model_id)
                results = evaluate_predictions(predictions)
                model_results[bench_name] = results
                with open(pred_path, "w") as f:
                    for p in predictions:
                        f.write(json.dumps(p) + "\n")
                print(format_results_table(results, f"{model_name} / {bench_name}"))
                print()
            all_results[model_name] = model_results
        save_results(all_results, output_dir / "all_results.json")
        logger.info(f"Judge results saved to {output_dir}")
        print_comparison_table(all_results)
        return

    # --aggregate-only: read existing predictions, recompute metrics, write all_results.json
    if args.aggregate_only:
        all_results = {}
        for model_cfg in config["evaluation"]["models"]:
            model_name = model_cfg["name"]
            if args.models and model_name not in args.models:
                continue
            model_results = {}
            for bench_cfg in config["evaluation"]["benchmarks"]:
                bench_name = bench_cfg["name"]
                if args.benchmarks and bench_name not in args.benchmarks:
                    continue
                pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
                if not pred_path.exists():
                    logger.warning(f"No predictions found: {pred_path}, skipping")
                    continue
                predictions = []
                with open(pred_path) as f:
                    for line in f:
                        if line.strip():
                            predictions.append(json.loads(line))
                logger.info(f"Aggregating {len(predictions)} predictions for {model_name}/{bench_name}")
                results = evaluate_predictions(predictions)
                model_results[bench_name] = results
                print(format_results_table(results, f"{model_name} / {bench_name}"))
                print()
            if model_results:
                all_results[model_name] = model_results
        save_results(all_results, output_dir / "all_results.json")
        logger.info(f"Aggregated results saved to {output_dir / 'all_results.json'}")
        print_comparison_table(all_results)
        return

    all_results = {}

    for model_cfg in config["evaluation"]["models"]:
        model_name = model_cfg["name"]
        if args.models and model_name not in args.models:
            continue

        use_mm = model_cfg.get("use_mm", False)

        if backend == "vllm" and use_mm:
            # Sequential loading: MM first (build memories), then AA (answer generation).
            # Avoids dual-model OOM — each model gets full gpu_memory_utilization.

            # Phase 1: Load MM, build memories for all conversations, save to disk, unload
            logger.info(f"Loading MM model (vLLM, sequential phase 1): {model_name}")
            mm_model = load_mm_model_vllm(
                model_cfg,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
            )
            if mm_model is not None:
                # Build memories for all benchmarks' conversations
                all_conv_sessions = {}
                for bench_cfg in config["evaluation"]["benchmarks"]:
                    bench_name = bench_cfg["name"]
                    if args.benchmarks and bench_name not in args.benchmarks:
                        continue
                    test_file = bench_cfg["test_file"]
                    if not Path(test_file).exists():
                        continue
                    test_data = load_test_data(test_file)
                    if args.max_examples and len(test_data) > args.max_examples:
                        test_data = test_data[:args.max_examples]
                    conv_groups = defaultdict(list)
                    for ex in test_data:
                        conv_groups[ex["conversation_id"]].append(ex)
                    for conv_id, examples in conv_groups.items():
                        if conv_id not in all_conv_sessions:
                            all_conv_sessions[conv_id] = examples[0].get("sessions", [])

                logger.info(f"Building memories for {len(all_conv_sessions)} conversations via MM...")
                prebuilt_memories = run_mm_all_conversations_vllm(mm_model, all_conv_sessions)

                # Save prebuilt memories to disk for crash recovery
                mem_cache_path = output_dir / f"{model_name}_mm_memories.json"
                with open(mem_cache_path, "w", encoding="utf-8") as f:
                    json.dump(prebuilt_memories, f)
                logger.info(f"Saved {len(prebuilt_memories)} conversation memories to {mem_cache_path}")

                # Unload MM
                del mm_model
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
            else:
                prebuilt_memories = None

            # Phase 2: Load AA at full GPU memory, run answer generation
            logger.info(f"Loading AA model (vLLM, sequential phase 2): {model_name}")
            model = load_model_vllm(
                model_cfg,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
            )
            tokenizer = None
            mm_model, mm_tokenizer = None, None

        elif backend == "vllm":
            logger.info(f"Loading model (vLLM): {model_name}")
            model = load_model_vllm(
                model_cfg,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
            )
            tokenizer = None
            mm_model, mm_tokenizer = None, None
            prebuilt_memories = None
        else:
            logger.info(f"Loading model (HF): {model_name}")
            model, tokenizer = load_model_and_tokenizer(model_cfg)
            mm_model, mm_tokenizer = None, None
            if use_mm:
                logger.info(f"Loading MM model (HF) for {model_name}")
                mm_model, mm_tokenizer = load_mm_model(model_cfg)
            prebuilt_memories = None

        model_results = {}

        for bench_cfg in config["evaluation"]["benchmarks"]:
            bench_name = bench_cfg["name"]
            if args.benchmarks and bench_name not in args.benchmarks:
                continue
            test_file = bench_cfg["test_file"]
            if not Path(test_file).exists():
                logger.warning(f"Test file not found: {test_file}, skipping")
                continue

            pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"

            # Check for existing complete predictions (resume support)
            expected_count = None
            if pred_path.exists():
                with open(pred_path) as f:
                    existing_count = sum(1 for line in f if line.strip())
                # Load benchmark size for comparison
                test_data_check = load_test_data(test_file)
                expected_count = min(len(test_data_check), args.max_examples) if args.max_examples else len(test_data_check)
                if existing_count >= expected_count:
                    logger.info(f"Skipping {model_name}/{bench_name} — {existing_count} predictions already exist")
                    # Load existing predictions for metrics
                    predictions = []
                    with open(pred_path) as f:
                        for line in f:
                            if line.strip():
                                predictions.append(json.loads(line))
                    results = evaluate_predictions(predictions)
                    model_results[bench_name] = results
                    print(format_results_table(results, f"{model_name} / {bench_name}"))
                    print()
                    continue
                else:
                    logger.info(f"Found {existing_count}/{expected_count} predictions for "
                                f"{model_name}/{bench_name}, re-running...")
                    # Remove partial file to avoid duplicates
                    pred_path.unlink()

            effective_mm_model = mm_model
            effective_mm_tokenizer = mm_tokenizer

            predictions = run_inference(
                model, tokenizer, model_cfg, bench_cfg,
                retrieval_top_k=retrieval_top_k,
                max_examples=args.max_examples,
                use_batched=not args.no_batch,
                inference_batch_size=inference_batch_size,
                max_new_tokens=max_new_tokens,
                mm_model=effective_mm_model,
                mm_tokenizer=effective_mm_tokenizer,
                backend=backend,
                output_path=str(pred_path),
                prebuilt_memories=prebuilt_memories,
            )

            # Run LLM judge if enabled
            if not args.skip_judge and "llm_judge" in config["evaluation"]["metrics"]:
                judge_cfg = config["evaluation"].get("llm_judge", {})
                predictions = judge_batch(
                    predictions,
                    model=judge_cfg.get("model", "gpt-4o-mini"),
                )

            results = evaluate_predictions(predictions)
            model_results[bench_name] = results

            # Overwrite with final version that includes judge scores (if any).
            with open(pred_path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")
            print(format_results_table(results, f"{model_name} / {bench_name}"))
            print()

        all_results[model_name] = model_results

        del model
        if tokenizer is not None:
            del tokenizer
        if mm_model is not None:
            del mm_model
        if mm_tokenizer is not None:
            del mm_tokenizer
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    save_results(all_results, output_dir / "all_results.json")
    logger.info(f"All results saved to {output_dir}")
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
