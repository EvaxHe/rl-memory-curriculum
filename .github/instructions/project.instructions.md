---
description: "Use when working on any part of the rl-memory-curriculum project. Covers project structure, dependency management, module conventions, and file I/O rules."
applyTo: "**"
---
# Project Conventions

## Structure

```
src/
  common/     # Shared utilities (config, prompts, scoring) — no ML imports
  memory/     # MemoryEntry, MemoryBank, retriever, heuristic
  agents/     # Answer Agent and Memory Manager (prompt building, parsing, execution)
  train/      # GRPO training entry point: python -m src.train.grpo
  eval/       # Evaluation entry point: python -m src.eval.runner
  pipeline.py # End-to-end inference
configs/      # YAML configs (train + eval)
tests/        # pytest-based tests
data/         # Data prep scripts + processed JSONL
```

## Dependency Management

- Use `uv` for everything — never pip/venv. Run scripts with `uv run python ...`.
- `pyproject.toml` is the single source of deps. `package = false` (not an installable package).
- ML-heavy optional deps (e.g. vllm) go in `[project.optional-dependencies]`, not main deps.
- Lazy-import heavy ML libraries (sentence-transformers, vllm) inside functions that need them, with try/except fallbacks.

## Module Execution

- Entry points use module syntax: `python -m src.train.grpo`, `python -m src.eval.runner`, `python -m src.eval.analyze`.
- Never use file-path execution (`python src/train/grpo.py`).
- Imports use dotted package paths: `from src.memory.bank import MemoryBank`.

## File I/O

- Always pass `encoding="utf-8"` to `open()` — the project runs on Windows where the default encoding is GBK.

## Config

- All training/eval hyperparameters live in YAML configs under `configs/`.
- Load configs with `src.common.config.load_config()`.
- No hardcoded hyperparameters in Python code.
