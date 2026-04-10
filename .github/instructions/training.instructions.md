---
description: "Use when modifying training code, reward functions, GRPO config, callbacks, or dataset preparation."
applyTo: "src/train/**"
---
# Training Code Conventions

## Architecture

- **Two-agent system**: Answer Agent (AA) and Memory Manager (MM), trained independently with GRPO.
- Training entry point: `python -m src.train.grpo --config <yaml> --agent answer_agent|memory_manager|both`
- Model loading via Unsloth `FastLanguageModel` (see `src/train/model.py`).
- Reward functions in `src/train/rewards.py` — AA uses token-level F1, MM uses format + quality rewards.

## Module Breakdown

| File | Responsibility |
|------|---------------|
| `grpo.py` | Entry point, `train_answer_agent()`, `train_memory_manager()`, CLI |
| `model.py` | `load_model_unsloth()` — Unsloth model + tokenizer setup |
| `rewards.py` | `make_aa_reward_func()`, `make_mm_quality_reward()`, format rewards |
| `datasets.py` | `prepare_aa_dataset()`, `prepare_mm_dataset()` from JSONL data |
| `callbacks.py` | TRL TrainerCallback subclasses for logging and checkpointing |

## Rules

- All hyperparameters come from YAML config — never hardcode learning rates, batch sizes, etc.
- Shared scoring logic (normalize_answer, token_f1, etc.) lives in `src/common/scoring.py`, not in training code.
- Shared prompt templates live in `src/common/prompts.py`.
- Checkpoint resume is built in — check for existing checkpoints before training.
