"""
Analyze results and generate tables/figures for the paper.

Produces:
1. Main results table (auto-detects models from results)
2. Per-question-type breakdown
3. AA-only vs AA+MM comparison (Phase 2)
4. Cost analysis table

Usage:
    python -m src.eval.analyze --results results/all_results.json --output paper/tables/
"""
import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "baseline_no_rl": "Qwen-2.5-7B (no RL)",
    "config_a_aa_only": "Config A AA-only (LoCoMo)",
    "config_b_aa_only": "Config B AA-only (Mixed)",
    "config_c_aa_only": "Config C AA-only (LME)",
    "config_a_full": "Config A AA+MM (LoCoMo)",
    "config_b_full": "Config B AA+MM (Mixed)",
    "config_c_full": "Config C AA+MM (LME)",
}


def load_results(path):
    with open(path) as f:
        return json.load(f)


def generate_main_table(results):
    preferred_order = [
        "baseline_no_rl",
        "config_a_aa_only", "config_a_full",
        "config_b_aa_only", "config_b_full",
        "config_c_aa_only", "config_c_full",
    ]
    model_keys = [k for k in preferred_order if k in results]
    for k in results:
        if k not in model_keys:
            model_keys.append(k)
    benchmarks = sorted({b for m in results.values() for b in m})
    lines = []
    lines.append("| Model | " + " | ".join(
        f"{b} F1 | {b} BLEU | {b} EM" for b in benchmarks) + " |")
    lines.append("|---" * (1 + len(benchmarks) * 3) + "|")
    for model_key in model_keys:
        label = MODEL_LABELS.get(model_key, model_key)
        row = [label]
        model_data = results.get(model_key, {})
        for bench in benchmarks:
            bench_data = model_data.get(bench, {}).get("overall", {})
            f1 = bench_data.get("f1", 0)
            bl = bench_data.get("bleu1", 0)
            em = bench_data.get("exact_match", 0)
            row.extend([f"{f1:.3f}", f"{bl:.3f}", f"{em:.3f}"])
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_per_type_table(results):
    config_candidates = {
        "A": ["config_a_full", "config_a_aa_only"],
        "B": ["config_b_full", "config_b_aa_only"],
        "C": ["config_c_full", "config_c_aa_only"],
    }
    models = []
    for label_prefix, candidates in config_candidates.items():
        for c in candidates:
            if c in results:
                models.append((c, MODEL_LABELS.get(c, c)))
                break
    if len(models) < 2:
        return "Insufficient models for per-type comparison."
    all_types = set()
    for model_key, _ in models:
        for bench_data in results.get(model_key, {}).values():
            all_types.update(bench_data.get("per_type", {}).keys())
    all_types = sorted(all_types)
    lines = []
    lines.append("| Question Type | " + " | ".join(
        label for _, label in models) + " | Delta (B-A) |")
    lines.append("|---" * (2 + len(models)) + "|")
    for qtype in all_types:
        row = [qtype]
        scores = []
        for model_key, _ in models:
            f1_sum, count = 0, 0
            for bench_data in results.get(model_key, {}).values():
                type_data = bench_data.get("per_type", {}).get(qtype, {})
                if type_data.get("n", 0) > 0:
                    f1_sum += type_data["f1"] * type_data["n"]
                    count += type_data["n"]
            f1 = f1_sum / count if count > 0 else 0
            scores.append(f1)
            row.append(f"{f1:.3f}")
        delta = scores[1] - scores[0] if len(scores) >= 2 else 0
        sign = "+" if delta >= 0 else ""
        row.append(f"{sign}{delta:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_aa_vs_mm_table(results):
    pairs = [
        ("config_a_aa_only", "config_a_full", "Config A (LoCoMo)"),
        ("config_b_aa_only", "config_b_full", "Config B (Mixed)"),
        ("config_c_aa_only", "config_c_full", "Config C (LME)"),
    ]
    has_pairs = any(aa in results and full in results for aa, full, _ in pairs)
    if not has_pairs:
        return ""
    benchmarks = sorted({b for m in results.values() for b in m})
    lines = []
    lines.append("| Config | Variant | " + " | ".join(
        f"{b} F1" for b in benchmarks) + " |")
    lines.append("|---" * (2 + len(benchmarks)) + "|")
    for aa_key, full_key, label in pairs:
        for variant_key, variant_label in [(aa_key, "AA-only"), (full_key, "AA+MM")]:
            if variant_key not in results:
                continue
            row = [label, variant_label]
            for bench in benchmarks:
                f1 = results[variant_key].get(bench, {}).get("overall", {}).get("f1", 0)
                row.append(f"{f1:.3f}")
            lines.append("| " + " | ".join(row) + " |")
        if aa_key in results and full_key in results:
            row = [label, "Delta (MM)"]
            for bench in benchmarks:
                aa_f1 = results[aa_key].get(bench, {}).get("overall", {}).get("f1", 0)
                full_f1 = results[full_key].get(bench, {}).get("overall", {}).get("f1", 0)
                delta = full_f1 - aa_f1
                sign = "+" if delta >= 0 else ""
                row.append(f"{sign}{delta:.3f}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def generate_cost_table(phase=1):
    if phase == 2:
        lines = [
            "| Component | Time | Cost |",
            "|---|---|---|",
            "| AA training (3 configs, 7B) | ~3h | varies |",
            "| MM training (3 configs, 7B) | ~3h | varies |",
            "| Eval (7 models x 2 benchmarks) | ~4h | varies |",
            "| LLM-as-Judge (~12K calls) | ~2.5h | ~$5.00 |",
            "| Setup + debug buffer | ~1h | varies |",
            "| **Total Phase 2** | **~13.5h** | **see README** |",
        ]
    else:
        lines = [
            "| Component | Time | Cost |",
            "|---|---|---|",
            "| Config A training (LoCoMo, 152 ex) | ~1h51m | ~$1.87 |",
            "| Config B training (Mixed, 212 ex) | ~2h39m | ~$2.67 |",
            "| Config C training (LME, 60 ex) | ~1h18m | ~$1.31 |",
            "| Evaluation (4 models x 2 benchmarks) | ~10h56m | ~$11.00 |",
            "| LLM-as-Judge (~5K calls) | ~1h39m | ~$3.00 |",
            "| Instance overhead (setup, debugging) | ~2h | ~$2.01 |",
            "| **Total Phase 1** | **~12h35m** | **~$12.60** |",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate paper tables")
    parser.add_argument("--results", type=str, default="results/all_results.json")
    parser.add_argument("--output", type=str, default="paper/tables/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = load_results(args.results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    table1 = generate_main_table(results)
    table2 = generate_per_type_table(results)
    table3 = generate_cost_table()

    for name, content in [
        ("table1_main_results.md", table1),
        ("table2_per_type.md", table2),
        ("table3_cost.md", table3),
    ]:
        path = output_dir / name
        with open(path, "w") as f:
            f.write(content)
        logger.info(f"Wrote {path}")


if __name__ == "__main__":
    main()
