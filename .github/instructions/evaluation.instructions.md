---
description: "Use when modifying evaluation code, inference, metrics, LLM judge, or results analysis."
applyTo: "src/eval/**"
---
# Evaluation Code Conventions

## Architecture

- Eval entry point: `python -m src.eval.runner --config configs/eval.yaml`
- Supports two inference backends: `hf` (HuggingFace Transformers) and `vllm` (vLLM offline engine).
- vLLM is an optional dependency — always lazy-import with try/except.

## Module Breakdown

| File | Responsibility |
|------|---------------|
| `runner.py` | Orchestrates eval: load models → inference → metrics → judge → analysis |
| `inference.py` | Batch inference with HF or vLLM backends |
| `model_loader.py` | Checkpoint detection and model/tokenizer loading |
| `metrics.py` | `evaluate_predictions()`, `format_results_table()`, `save_results()` |
| `judge.py` | LLM-as-Judge via OpenAI-compatible API |
| `analyze.py` | Generate paper tables from `all_results.json` |

## Rules

- Metrics (F1, BLEU-1, EM) use `src/common/scoring.py` — do not reimplement.
- Answer extraction uses `src/agents/answer_agent.extract_answer_from_completion()`.
- All eval config (model paths, batch sizes, benchmarks) comes from `configs/eval.yaml`.
- Results go to `results/` directory. Paper tables go to `paper/tables/`.
