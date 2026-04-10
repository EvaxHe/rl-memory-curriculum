"""
Evaluation metrics for Memory-R1 experiments.

Computes F1, BLEU-1, Exact Match per example and aggregates
by question type and benchmark.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path

from src.common.scoring import token_f1, bleu1, exact_match

logger = logging.getLogger(__name__)


def evaluate_predictions(predictions: list[dict]) -> dict:
    """
    Compute metrics over a list of predictions.

    Each prediction dict must have:
    - "answer": model's predicted answer
    - "gold_answer": ground truth
    - "question_type": category (for per-type breakdown)
    - "source_benchmark": which benchmark this came from
    """
    # Per-example metrics
    for pred in predictions:
        pred_ans = pred.get("answer", "")
        gold_ans = pred.get("gold_answer", "")
        pred["metrics"] = {
            "f1": token_f1(pred_ans, gold_ans),
            "bleu1": bleu1(pred_ans, gold_ans),
            "exact_match": exact_match(pred_ans, gold_ans),
        }

    # Aggregate: overall
    overall = aggregate_metrics(predictions)

    # Aggregate: by question type
    by_type = defaultdict(list)
    for pred in predictions:
        by_type[pred.get("question_type", "unknown")].append(pred)
    per_type = {t: aggregate_metrics(preds) for t, preds in by_type.items()}

    # Aggregate: by benchmark
    by_bench = defaultdict(list)
    for pred in predictions:
        by_bench[pred.get("source_benchmark", "unknown")].append(pred)
    per_bench = {b: aggregate_metrics(preds) for b, preds in by_bench.items()}

    return {
        "overall": overall,
        "per_type": per_type,
        "per_benchmark": per_bench,
        "num_examples": len(predictions),
    }


def aggregate_metrics(predictions: list[dict]) -> dict:
    """Compute mean metrics over a list of predictions."""
    if not predictions:
        return {"f1": 0.0, "bleu1": 0.0, "exact_match": 0.0, "n": 0}

    metrics = defaultdict(float)
    for pred in predictions:
        for k, v in pred.get("metrics", {}).items():
            metrics[k] += v

    n = len(predictions)
    return {k: v / n for k, v in metrics.items()} | {"n": n}


def format_results_table(results: dict, model_name: str) -> str:
    """Format results as a markdown table for the paper."""
    lines = [f"### {model_name}\n"]

    # Overall
    o = results["overall"]
    lines.append(f"**Overall** (n={o['n']}): F1={o['f1']:.3f}, "
                 f"BLEU-1={o['bleu1']:.3f}, EM={o['exact_match']:.3f}\n")

    # Per type
    lines.append("| Question Type | N | F1 | BLEU-1 | EM |")
    lines.append("|---|---|---|---|---|")
    for qtype, m in sorted(results["per_type"].items()):
        lines.append(f"| {qtype} | {m['n']} | {m['f1']:.3f} | "
                     f"{m['bleu1']:.3f} | {m['exact_match']:.3f} |")

    return "\n".join(lines)


def save_results(results: dict, output_path: Path):
    """Save results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")
