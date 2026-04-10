"""
Run evaluation across all configs and benchmarks.

Behavior controlled by eval YAML config.
Supports two inference backends:
  - "hf"   : HuggingFace Transformers model.generate() with manual batching
  - "vllm" : vLLM offline LLM engine with continuous batching + PagedAttention

Usage:
    python eval/run_eval.py --config configs/eval.yaml
    python eval/run_eval.py --config configs/eval.yaml --backend vllm
    python eval/run_eval.py --config configs/eval.yaml --backend vllm --gpus 4
    python eval/run_eval.py --config configs/eval.yaml --skip-judge
    python eval/run_eval.py --config configs/eval.yaml --judge-only
    python eval/run_eval.py --config configs/eval.yaml --aggregate-only
    python eval/run_eval.py --config configs/eval.yaml --models config_a_full
"""
import argparse
import gc
import json
import logging
import re
import sys
import tempfile
import yaml
from pathlib import Path
from collections import defaultdict

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import evaluate_predictions, format_results_table, save_results
from eval.judge import judge_batch

logger = logging.getLogger(__name__)


# ============================================================
# Model loading (torch imported lazily - GPU only)
# ============================================================

def load_model_and_tokenizer(model_config):
    """Load model + tokenizer. Handles base model, LoRA, and full FT checkpoints."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = model_config["checkpoint"]
    # Auto-detect checkpoint type from training_meta.json if available
    is_lora = model_config.get("lora", False)
    is_full_ft = model_config.get("full_ft", False)

    if not is_lora and not is_full_ft and not model_config.get("is_baseline", False):
        # Auto-detect from training_meta.json in checkpoint dir
        meta_path = Path(checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"Auto-detected from training_meta.json: lora={is_lora}, full_ft={is_full_ft}")

    if is_lora:
        from peft import PeftModel
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        logger.info(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        logger.info(f"Loading LoRA adapter: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    elif is_full_ft:
        # Full fine-tuned checkpoint: load directly from saved directory
        logger.info(f"Loading full FT checkpoint: {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    else:
        # Base model (no training)
        logger.info(f"Loading base model: {checkpoint}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


# ============================================================
# Heuristic memory builder (no GPU needed)
# ============================================================

SKIP_WORDS = {"hi", "hello", "hey", "thanks", "bye", "ok", "okay", "yes", "no"}


def build_heuristic_memories(sessions, max_memories=200):
    """Build memory entries from sessions using heuristics."""
    memories = []
    for session in sessions:
        sid = session.get("session_id", 0)
        dt = session.get("date_time", "")
        for turn in session.get("turns", []):
            text = turn.get("text", "").strip()
            speaker = turn.get("speaker", "")
            words = text.lower().split()
            if len(words) > 5 and not any(w in SKIP_WORDS for w in words[:2]):
                mem = f"{speaker}: {text[:300]}"
                if dt:
                    mem += f" (session {sid}, {dt})"
                memories.append(mem)
    if len(memories) > max_memories:
        step = len(memories) / max_memories
        memories = [memories[int(i * step)] for i in range(max_memories)]
    return memories


def retrieve_memories(question, memories, top_k=20):
    """Retrieve top-k memories. Embedding search with keyword fallback."""
    if not memories:
        return []
    try:
        from src.retriever import embed_texts, search_numpy_fallback
        corpus_emb = embed_texts(memories)
        if corpus_emb is not None:
            query_emb = embed_texts([question])
            if query_emb is not None:
                _, indices = search_numpy_fallback(
                    query_emb[0], corpus_emb, top_k=min(top_k, len(memories)),
                )
                return [memories[i] for i in indices if i < len(memories)]
    except Exception:
        pass
    q_words = set(question.lower().split())
    scored = [(len(q_words & set(m.lower().split())), m) for m in memories]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


# ============================================================
# Answer generation (GPU) — single and batched
# ============================================================

AA_SYSTEM_PROMPT = """You are an Answer Agent for a conversational AI assistant.
You have access to a memory bank containing facts from past conversations.

Given a question and retrieved memories, you must:
1. Select the most relevant memories for answering the question.
2. Reason step-by-step using the selected memories.
3. Provide a concise, accurate answer.

Output format:
<selected_memories>
[list the memory IDs or snippets you're using]
</selected_memories>
<reasoning>
[your step-by-step reasoning]
</reasoning>
<answer>
[your final answer - be concise]
</answer>"""

AA_USER_TEMPLATE = """## Retrieved Memories (top {num_retrieved})
{memories}

## Question
{question}

## Your Response:"""


def format_aa_prompt(question, retrieved_memories):
    """Format a single AA prompt (returns chat messages list)."""
    mem_str = "\n".join(f"- {m}" for m in retrieved_memories) \
        if retrieved_memories else "No relevant memories found."
    return [
        {"role": "system", "content": AA_SYSTEM_PROMPT},
        {"role": "user", "content": AA_USER_TEMPLATE.format(
            num_retrieved=len(retrieved_memories),
            memories=mem_str, question=question,
        )},
    ]


def extract_answer(raw_output):
    """Extract answer from model output."""
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_output, re.DOTALL)
    if ans_match:
        return ans_match.group(1).strip()
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    return lines[-1] if lines else raw_output.strip()


def generate_answer(model, tokenizer, question, retrieved_memories,
                    max_new_tokens=512):
    """Generate an answer for a single question (fallback for non-batched)."""
    import torch
    messages = format_aa_prompt(question, retrieved_memories)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_answer(raw_output)


def generate_answers_batched(model, tokenizer, prompts_list,
                             max_new_tokens=1024, batch_size=8):
    """
    Batched answer generation. Significantly faster than one-at-a-time.
    prompts_list: list of chat message lists (one per question).
    Returns list of extracted answer strings.
    """
    import torch

    # Convert chat messages to text
    texts = []
    for messages in prompts_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    answers = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.3, do_sample=True, top_p=0.9,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            new_tokens = output[input_len:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers.append(extract_answer(raw_output))

    return answers


# ============================================================
# vLLM answer generation
# ============================================================

def generate_answers_vllm(llm, prompts_list, max_new_tokens=1024,
                          temperature=0.3, top_p=0.9):
    """
    Generate answers for all prompts using vLLM offline engine.
    vLLM handles continuous batching + PagedAttention internally.

    prompts_list: list of chat message lists (one per question).
    Returns list of extracted answer strings.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    texts = []
    for messages in prompts_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    logger.info(f"vLLM generating {len(texts)} prompts...")
    outputs = llm.generate(texts, sampling_params)

    answers = []
    for output in outputs:
        raw_output = output.outputs[0].text
        answers.append(extract_answer(raw_output))

    return answers


def run_mm_on_sessions_vllm(mm_llm, sessions, max_new_tokens=256):
    """
    Run trained Memory Manager on all dialogue turns using vLLM.
    Sequential per conversation (each turn depends on memory state from prior turns).
    Returns list of memory strings.
    """
    from vllm import SamplingParams
    from src.memory_bank import MemoryBank
    from src.memory_manager import build_mm_prompt, parse_mm_output, execute_mm_operation

    tokenizer = mm_llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.3, top_p=0.9, max_tokens=max_new_tokens,
    )
    bank = MemoryBank(use_embeddings=False)

    for session in sessions:
        sid = session.get("session_id", 0)
        for i, turn in enumerate(session.get("turns", [])):
            messages = build_mm_prompt(
                bank, session_id=sid, turn_id=i,
                speaker=turn["speaker"], message=turn["text"][:500],
            )
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            outputs = mm_llm.generate([text], sampling_params)
            raw_output = outputs[0].outputs[0].text

            operation = parse_mm_output(raw_output)
            execute_mm_operation(operation, bank, session_id=sid)
            bank.advance_turn()

    return [e.content for e in bank.get_all()]


def run_mm_all_conversations_vllm(mm_llm, conv_sessions, max_new_tokens=256):
    """
    Batch MM processing across all conversations using vLLM.

    Instead of processing each conversation sequentially (batch_size=1 per vLLM call),
    this processes all conversations step-wise: at each step, gather one pending turn
    from every active conversation and send them as a single batch to vLLM.

    conv_sessions: dict of {conv_id: sessions_list}
    Returns: dict of {conv_id: list[str]} (memory strings per conversation)
    """
    from vllm import SamplingParams
    from src.memory_bank import MemoryBank
    from src.memory_manager import build_mm_prompt, parse_mm_output, execute_mm_operation

    tokenizer = mm_llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.3, top_p=0.9, max_tokens=max_new_tokens,
    )

    # Build per-conversation state: bank + flattened turn queue
    conv_banks = {}
    conv_turn_queues = {}
    for conv_id, sessions in conv_sessions.items():
        conv_banks[conv_id] = MemoryBank(use_embeddings=False)
        turns = []
        for session in sessions:
            sid = session.get("session_id", 0)
            for i, turn in enumerate(session.get("turns", [])):
                turns.append((sid, i, turn))
        conv_turn_queues[conv_id] = turns

    # Track position per conversation
    conv_positions = {conv_id: 0 for conv_id in conv_turn_queues}
    total_turns = sum(len(q) for q in conv_turn_queues.values())
    processed = 0
    step = 0

    while True:
        # Gather one pending turn from each active conversation
        batch_conv_ids = []
        batch_texts = []
        batch_sids = []

        for conv_id, pos in conv_positions.items():
            queue = conv_turn_queues[conv_id]
            if pos >= len(queue):
                continue
            sid, turn_idx, turn = queue[pos]
            bank = conv_banks[conv_id]
            messages = build_mm_prompt(
                bank, session_id=sid, turn_id=turn_idx,
                speaker=turn["speaker"], message=turn["text"][:500],
            )
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            batch_conv_ids.append(conv_id)
            batch_texts.append(text)
            batch_sids.append(sid)

        if not batch_texts:
            break

        step += 1
        if step % 10 == 1 or len(batch_texts) > 1:
            logger.info(
                f"  MM batch step {step}: {len(batch_texts)} prompts "
                f"({processed}/{total_turns} turns done)"
            )

        outputs = mm_llm.generate(batch_texts, sampling_params)

        for conv_id, sid, output in zip(batch_conv_ids, batch_sids, outputs):
            raw_output = output.outputs[0].text
            bank = conv_banks[conv_id]
            operation = parse_mm_output(raw_output)
            execute_mm_operation(operation, bank, session_id=sid)
            bank.advance_turn()
            conv_positions[conv_id] += 1
            processed += 1

    logger.info(f"  MM batched processing complete: {processed} turns in {step} steps")

    return {
        conv_id: [e.content for e in bank.get_all()]
        for conv_id, bank in conv_banks.items()
    }


# ============================================================
# Inference orchestration
# ============================================================

def load_test_data(test_file):
    """Load test JSONL file."""
    data = []
    with open(test_file) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_mm_model(model_config):
    """Load Memory Manager model for use_mm inference. Returns (model, tokenizer) or (None, None)."""
    mm_checkpoint = model_config.get("mm_checkpoint")
    if not mm_checkpoint:
        logger.warning("use_mm=true but no mm_checkpoint specified, falling back to heuristic")
        return None, None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_full_ft = model_config.get("full_ft", False)
    is_lora = model_config.get("lora", False)

    if not is_lora and not is_full_ft:
        # Auto-detect from training_meta.json in MM checkpoint dir
        meta_path = Path(mm_checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"MM auto-detected: lora={is_lora}, full_ft={is_full_ft}")

    if is_full_ft:
        logger.info(f"Loading MM full FT checkpoint: {mm_checkpoint}")
        mm_model = AutoModelForCausalLM.from_pretrained(
            mm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)
    elif is_lora:
        from peft import PeftModel
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        mm_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_model = PeftModel.from_pretrained(mm_model, mm_checkpoint)
        mm_model = mm_model.merge_and_unload()
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)
    else:
        logger.info(f"Loading MM base model: {mm_checkpoint}")
        mm_model = AutoModelForCausalLM.from_pretrained(
            mm_checkpoint, torch_dtype=torch.bfloat16, device_map="auto",
        )
        mm_tokenizer = AutoTokenizer.from_pretrained(mm_checkpoint)

    if mm_tokenizer.pad_token is None:
        mm_tokenizer.pad_token = mm_tokenizer.eos_token
    mm_model.eval()
    return mm_model, mm_tokenizer


# ============================================================
# vLLM model loading
# ============================================================

def _detect_checkpoint_type(checkpoint, model_config):
    """Auto-detect LoRA vs full FT from training_meta.json. Returns (is_lora, is_full_ft)."""
    is_lora = model_config.get("lora", False)
    is_full_ft = model_config.get("full_ft", False)
    if not is_lora and not is_full_ft and not model_config.get("is_baseline", False):
        meta_path = Path(checkpoint) / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("use_lora", False):
                is_lora = True
            elif meta.get("use_lora") is False:
                is_full_ft = True
            logger.info(f"Auto-detected: lora={is_lora}, full_ft={is_full_ft}")
    return is_lora, is_full_ft


def _merge_lora_to_tmpdir(checkpoint, base_model_name):
    """Merge LoRA adapter into base model and save to a temp directory for vLLM."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Merging LoRA {checkpoint} into {base_model_name} for vLLM...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16, device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base, checkpoint)
    merged = merged.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tmpdir = tempfile.mkdtemp(prefix="vllm_merged_")
    merged.save_pretrained(tmpdir)
    tokenizer.save_pretrained(tmpdir)
    del base, merged
    gc.collect()
    logger.info(f"Merged LoRA checkpoint saved to {tmpdir}")
    return tmpdir


def load_model_vllm(model_config, gpu_memory_utilization=0.85,
                     tensor_parallel_size=1, max_model_len=4096):
    """Load model via vLLM offline LLM engine. Returns vllm.LLM instance."""
    from vllm import LLM

    checkpoint = model_config["checkpoint"]
    is_lora, is_full_ft = _detect_checkpoint_type(checkpoint, model_config)

    if is_lora:
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        model_path = _merge_lora_to_tmpdir(checkpoint, base_model_name)
    else:
        model_path = checkpoint

    logger.info(f"Loading vLLM model: {model_path} (tp={tensor_parallel_size}, "
                f"gpu_mem={gpu_memory_utilization})")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    return llm


def load_mm_model_vllm(model_config, gpu_memory_utilization=0.45,
                        tensor_parallel_size=1, max_model_len=4096):
    """Load Memory Manager model via vLLM. Returns vllm.LLM instance or None."""
    from vllm import LLM

    mm_checkpoint = model_config.get("mm_checkpoint")
    if not mm_checkpoint:
        logger.warning("use_mm=true but no mm_checkpoint specified, falling back to heuristic")
        return None

    is_lora, is_full_ft = _detect_checkpoint_type(mm_checkpoint, model_config)

    if is_lora:
        base_model_name = model_config.get("base_model", "unsloth/Qwen3.5-4B")
        model_path = _merge_lora_to_tmpdir(mm_checkpoint, base_model_name)
    else:
        model_path = mm_checkpoint

    logger.info(f"Loading vLLM MM model: {model_path} (tp={tensor_parallel_size}, "
                f"gpu_mem={gpu_memory_utilization})")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )
    return llm


def run_mm_on_sessions(mm_model, mm_tokenizer, sessions, max_new_tokens=256):
    """
    Run trained Memory Manager on all dialogue turns to build a memory bank.
    Returns list of memory strings (same format as heuristic memories).
    """
    import torch
    from src.memory_bank import MemoryBank
    from src.memory_manager import build_mm_prompt, parse_mm_output, execute_mm_operation

    bank = MemoryBank(use_embeddings=False)

    for session in sessions:
        sid = session.get("session_id", 0)
        for i, turn in enumerate(session.get("turns", [])):
            messages = build_mm_prompt(
                bank, session_id=sid, turn_id=i,
                speaker=turn["speaker"], message=turn["text"][:500],
            )
            text = mm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = mm_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=4096,
            ).to(mm_model.device)

            with torch.no_grad():
                outputs = mm_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.3, do_sample=True, top_p=0.9,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = mm_tokenizer.decode(new_tokens, skip_special_tokens=True)

            operation = parse_mm_output(raw_output)
            execute_mm_operation(operation, bank, session_id=sid)
            bank.advance_turn()

    # Convert bank entries to string list (same format as heuristic)
    return [e.content for e in bank.get_all()]


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
    If prebuilt_memories is provided (dict of conv_id → list[str]), uses those instead of running MM.
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


# ============================================================
# Config + Main
# ============================================================

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


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
    from dotenv import load_dotenv
    load_dotenv()

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
            # Pass prebuilt memories via a temporary attribute on model_cfg
            # so run_inference can use them instead of re-running MM
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

            # If we have prebuilt memories from sequential vLLM MM phase,
            # inject them so run_inference uses them instead of re-running MM
            effective_mm_model = mm_model
            effective_mm_tokenizer = mm_tokenizer
            if prebuilt_memories is not None:
                # Temporarily disable use_mm in model_cfg so run_inference
                # uses heuristic path, but we'll override conv_memories below
                pass

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

            # Predictions were written incrementally by run_inference.
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

    # Log evaluation results to wandb
    wandb_cfg = config.get("wandb", {})
    if wandb_cfg.get("enabled", False) and all_results:
        _log_eval_to_wandb(all_results, config, output_dir, wandb_cfg)


def _log_eval_to_wandb(all_results, config, output_dir, wandb_cfg):
    """Log evaluation metrics, comparison table, and artifacts to wandb."""
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping eval logging")
        return

    project = wandb_cfg.get("project", "rl-memory-curriculum")
    entity = wandb_cfg.get("entity")

    run = wandb.init(
        project=project,
        entity=entity,
        job_type="eval",
        name="evaluation",
        config={"eval_config": config["evaluation"]},
    )

    # Log per-model, per-benchmark scalar metrics
    for model_name, model_results in all_results.items():
        for bench_name, results in model_results.items():
            overall = results.get("overall", {})
            prefix = f"{model_name}/{bench_name}"
            run.summary[f"{prefix}/f1"] = overall.get("f1", 0)
            run.summary[f"{prefix}/bleu1"] = overall.get("bleu1", 0)
            run.summary[f"{prefix}/exact_match"] = overall.get("exact_match", 0)
            run.summary[f"{prefix}/n"] = overall.get("n", 0)

            # Judge score if available
            if "judge_score" in overall:
                run.summary[f"{prefix}/judge_score"] = overall["judge_score"]

    # Build comparison table
    table_columns = ["Model", "Benchmark", "F1", "BLEU-1", "EM", "N"]
    table_data = []
    for model_name, model_results in all_results.items():
        for bench_name, results in model_results.items():
            o = results.get("overall", {})
            table_data.append([
                model_name, bench_name,
                round(o.get("f1", 0), 4),
                round(o.get("bleu1", 0), 4),
                round(o.get("exact_match", 0), 4),
                o.get("n", 0),
            ])
    if table_data:
        wandb_table = wandb.Table(columns=table_columns, data=table_data)
        run.log({"eval/comparison_table": wandb_table})

    # Build per-question-type table
    type_columns = ["Model", "Benchmark", "Question Type", "F1", "BLEU-1", "EM", "N"]
    type_data = []
    for model_name, model_results in all_results.items():
        for bench_name, results in model_results.items():
            for qtype, metrics in results.get("per_type", {}).items():
                type_data.append([
                    model_name, bench_name, qtype,
                    round(metrics.get("f1", 0), 4),
                    round(metrics.get("bleu1", 0), 4),
                    round(metrics.get("exact_match", 0), 4),
                    metrics.get("n", 0),
                ])
    if type_data:
        type_table = wandb.Table(columns=type_columns, data=type_data)
        run.log({"eval/per_type_table": type_table})

    # Log all_results.json as artifact
    artifact = wandb.Artifact("eval-results", type="eval-results")
    results_path = output_dir / "all_results.json"
    if results_path.exists():
        artifact.add_file(str(results_path))
    run.log_artifact(artifact)

    run.finish()
    logger.info("Evaluation results logged to wandb")


if __name__ == "__main__":
    main()
