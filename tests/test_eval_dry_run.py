"""
Local dry-run of the full eval pipeline with dummy predictions.

Generates fake predictions for all 7 models × 2 benchmarks,
runs metrics, judge (mocked), and analysis to verify tables render.

Also tests vLLM code paths (with mocked vLLM imports).

Usage:
    python -m pytest tests/test_eval_dry_run.py -v
    python tests/test_eval_dry_run.py  # standalone
"""
import json
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import evaluate_predictions, format_results_table, save_results
from eval.analyze_results import (
    generate_main_table, generate_per_type_table, generate_cost_table,
    generate_aa_vs_mm_table,
)


# ============================================================
# Dummy prediction generator
# ============================================================

MODELS = [
    "baseline_no_rl",
    "config_a_aa_only",
    "config_b_aa_only",
    "config_c_aa_only",
    "config_a_full",
    "config_b_full",
    "config_c_full",
]

_PROJECT_ROOT = Path(__file__).parent.parent

BENCHMARKS = {
    "locomo": str(_PROJECT_ROOT / "data/processed/locomo_test.jsonl"),
    "longmemeval": str(_PROJECT_ROOT / "data/processed/longmemeval_test.jsonl"),
}

MODEL_QUALITY = {
    "baseline_no_rl": 0.25,
    "config_a_aa_only": 0.60,
    "config_b_aa_only": 0.65,
    "config_c_aa_only": 0.50,
    "config_a_full": 0.70,
    "config_b_full": 0.75,
    "config_c_full": 0.55,
}


def make_dummy_answer(gold_answer: str, quality: float) -> str:
    """Generate a dummy predicted answer with controlled overlap to gold."""
    words = gold_answer.split()
    if not words:
        return "unknown"
    result = []
    for w in words:
        if random.random() < quality:
            result.append(w)
        else:
            result.append("something")
    if random.random() < 0.3:
        result.append("additionally")
    return " ".join(result) if result else "unknown"


def generate_dummy_predictions(test_file: str, model_name: str,
                               benchmark_name: str, max_examples=None):
    """Read real test data, generate dummy predictions."""
    quality = MODEL_QUALITY.get(model_name, 0.3)
    predictions = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            if max_examples and i >= max_examples:
                break
            ex = json.loads(line)
            pred_answer = make_dummy_answer(ex["answer"], quality)
            predictions.append({
                "question": ex["question"],
                "gold_answer": ex["answer"],
                "answer": pred_answer,
                "question_type": ex.get("question_type", "unknown"),
                "source_benchmark": ex.get("source_benchmark", benchmark_name),
                "model": model_name,
                "conversation_id": ex.get("conversation_id", "unknown"),
                "judge_score": random.randint(1, 5),
                "judge_reason": "mock evaluation",
            })
    return predictions


# ============================================================
# Tests
# ============================================================

def test_metrics_on_dummy_predictions():
    """Verify metrics compute correctly on dummy data."""
    random.seed(42)
    preds = generate_dummy_predictions(
        BENCHMARKS["locomo"], "config_b_full", "locomo", max_examples=50
    )
    results = evaluate_predictions(preds)

    assert "overall" in results
    assert "per_type" in results
    assert results["num_examples"] == 50
    assert 0 <= results["overall"]["f1"] <= 1
    assert 0 <= results["overall"]["bleu1"] <= 1
    assert 0 <= results["overall"]["exact_match"] <= 1
    assert results["overall"]["n"] == 50

    assert len(results["per_type"]) > 0
    for qtype, m in results["per_type"].items():
        assert m["n"] > 0
        assert 0 <= m["f1"] <= 1


def test_format_results_table():
    """Verify table formatting doesn't crash."""
    random.seed(42)
    preds = generate_dummy_predictions(
        BENCHMARKS["locomo"], "config_a_full", "locomo", max_examples=30
    )
    results = evaluate_predictions(preds)
    table = format_results_table(results, "config_a / locomo")
    assert "config_a / locomo" in table
    assert "F1" in table
    assert "|" in table


def test_full_eval_pipeline_dry_run():
    """
    Full dry-run: generate predictions for all 7 models × 2 benchmarks,
    compute metrics, save results, generate paper tables.
    """
    random.seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "results"
        output_dir.mkdir()
        tables_dir = Path(tmpdir) / "paper" / "tables"
        tables_dir.mkdir(parents=True)

        all_results = {}
        for model_name in MODELS:
            model_results = {}
            for bench_name, test_file in BENCHMARKS.items():
                preds = generate_dummy_predictions(
                    test_file, model_name, bench_name, max_examples=100
                )
                results = evaluate_predictions(preds)
                model_results[bench_name] = results

                pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
                with open(pred_path, "w") as f:
                    for p in preds:
                        f.write(json.dumps(p) + "\n")

                table = format_results_table(results, f"{model_name} / {bench_name}")
                print(table)
                print()

            all_results[model_name] = model_results

        # Save results
        results_path = output_dir / "all_results.json"
        save_results(all_results, results_path)
        assert results_path.exists()

        with open(results_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 7  # 7 models

        # Generate tables
        table1 = generate_main_table(loaded)
        table2 = generate_per_type_table(loaded)
        table3 = generate_cost_table(phase=2)
        table4 = generate_aa_vs_mm_table(loaded)

        print("\n" + "=" * 80)
        print("TABLE 1: Main Results")
        print("=" * 80)
        print(table1)
        print("\n" + "=" * 80)
        print("TABLE 2: Per-Question-Type F1")
        print("=" * 80)
        print(table2)
        print("\n" + "=" * 80)
        print("TABLE 3: Cost")
        print("=" * 80)
        print(table3)
        print("\n" + "=" * 80)
        print("TABLE 4: AA-only vs AA+MM")
        print("=" * 80)
        print(table4)

        # Save tables
        for name, content in [
            ("table1_main_results.md", table1),
            ("table2_per_type.md", table2),
            ("table3_cost.md", table3),
            ("table4_aa_vs_mm.md", table4),
        ]:
            path = tables_dir / name
            with open(path, "w") as f:
                f.write(content)
            assert path.exists()

        # Sanity checks
        assert "Config A" in table1 or "config_a" in table1
        assert "Config B" in table1 or "config_b" in table1
        assert "|" in table1
        assert "|" in table2
        assert "Total" in table3 or "total" in table3.lower()
        assert "AA-only" in table4
        assert "AA+MM" in table4

        # Verify prediction files
        for model_name in MODELS:
            for bench_name in BENCHMARKS:
                pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
                assert pred_path.exists()
                with open(pred_path) as f:
                    lines = f.readlines()
                assert len(lines) > 0
                first = json.loads(lines[0])
                assert "question" in first
                assert "gold_answer" in first
                assert "answer" in first
                assert "question_type" in first

        print("\n✓ Full eval pipeline dry-run PASSED")


def test_vllm_generate_answers():
    """Test vLLM answer generation code path with mocked vLLM."""
    from eval.run_eval import extract_answer, format_aa_prompt

    # Mock vLLM output object
    mock_output_obj = MagicMock()
    mock_output_obj.outputs = [MagicMock(text="<answer>Paris</answer>")]

    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
    mock_llm.get_tokenizer.return_value = mock_tokenizer
    mock_llm.generate.return_value = [mock_output_obj, mock_output_obj]

    mock_sampling_params = MagicMock()

    with patch.dict("sys.modules", {
        "vllm": MagicMock(SamplingParams=lambda **kwargs: mock_sampling_params),
    }):
        from eval.run_eval import generate_answers_vllm
        prompts = [
            format_aa_prompt("What is the capital of France?", ["France is in Europe"]),
            format_aa_prompt("Who wrote Hamlet?", ["Shakespeare was a playwright"]),
        ]
        answers = generate_answers_vllm(mock_llm, prompts)
        assert len(answers) == 2
        assert answers[0] == "Paris"
        mock_llm.generate.assert_called_once()


def test_vllm_extract_answer_fallback():
    """Test extract_answer handles non-tagged vLLM output gracefully."""
    from eval.run_eval import extract_answer
    assert extract_answer("<answer>42</answer>") == "42"
    assert extract_answer("The answer is 42") == "The answer is 42"
    assert extract_answer("line1\nline2\nline3") == "line3"
    assert extract_answer("") == ""


def test_detect_checkpoint_type():
    """Test _detect_checkpoint_type auto-detection from config flags."""
    from eval.run_eval import _detect_checkpoint_type

    # Explicit lora flag
    is_lora, is_full_ft = _detect_checkpoint_type(
        "/nonexistent", {"lora": True}
    )
    assert is_lora is True
    assert is_full_ft is False

    # Explicit full_ft flag
    is_lora, is_full_ft = _detect_checkpoint_type(
        "/nonexistent", {"full_ft": True}
    )
    assert is_lora is False
    assert is_full_ft is True

    # Baseline
    is_lora, is_full_ft = _detect_checkpoint_type(
        "/nonexistent", {"is_baseline": True}
    )
    assert is_lora is False
    assert is_full_ft is False


if __name__ == "__main__":
    test_metrics_on_dummy_predictions()
    print("✓ test_metrics_on_dummy_predictions passed")
    test_format_results_table()
    print("✓ test_format_results_table passed")
    test_full_eval_pipeline_dry_run()
    test_vllm_generate_answers()
    print("✓ test_vllm_generate_answers passed")
    test_vllm_extract_answer_fallback()
    print("✓ test_vllm_extract_answer_fallback passed")
    test_detect_checkpoint_type()
    print("✓ test_detect_checkpoint_type passed")
    print("\n✓ All eval dry-run tests passed")
