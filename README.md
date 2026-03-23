# rl-memory-curriculum

Does mixed-benchmark training improve RL-based external memory management for LLM agents?

## What This Is

LLM agents that maintain long-term conversations need an external memory — a structured
store of facts from past sessions that the agent can read, write, update, and delete.
[Memory-R1](https://arxiv.org/abs/2508.19828) showed that training this memory policy
with RL (GRPO) significantly outperforms heuristic approaches.

Memory-R1 trains on 152 QA pairs from a single benchmark (LoCoMo). We ask: does
training on a more diverse mix of benchmarks improve generalization?

This repo implements the Memory-R1 architecture (no public code exists) and runs
a controlled curriculum experiment across three training configurations.

## Research Question

1. Does training on a mixed curriculum (LoCoMo + LongMemEval) improve cross-benchmark generalization?
2. Does training on LongMemEval alone teach different memory skills?
3. Which question types benefit most from curriculum diversity?

## Quick Start

```bash
bash scripts/setup.sh           # create uv-managed venv, sync deps, download tokenizer
bash scripts/run_all.sh         # train 3 configs, evaluate, generate tables
```

Dependencies are managed with `uv` via `pyproject.toml`.
Torch is configured to use CUDA wheels (cu130) on Linux/Windows x86_64.

To verify GPU runtime after setup:
```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Raw and processed benchmark data are not committed to GitHub. To prepare data locally:
```bash
git clone https://github.com/snap-research/locomo data/raw/locomo
git clone https://github.com/xiaowu0162/LongMemEval data/raw/longmemeval

# LongMemEval repo clone does not include the benchmark json by default.
curl -L -o data/raw/longmemeval/data/longmemeval_s_cleaned.json \
	https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

python data/prepare_locomo.py
python data/prepare_longmemeval.py
python data/prepare_mixed.py
```
If you use the project environment, run these via `uv run python ...`.
The setup and run scripts detect local processed files automatically once they exist.

## Hardware Requirements

| Tier | GPU | VRAM | Config | Est. Time | Est. Cost (Lambda) |
|------|-----|------|--------|-----------|-------------------|
| Default (LoRA) | L40S, A100 40GB+ | ≥48 GB | `configs/` | ~10h | ~$15-30 |
| Full FT | H100, A100 80GB | ≥80 GB | `configs/full_ft/` | ~14h | ~$40-80 |

To use full fine-tuning configs:
```bash
bash scripts/run_all.sh --config-dir configs/full_ft
```

## Experimental Configurations

| Config | Training Data | Size |
|--------|--------------|------|
| A (baseline) | LoCoMo only | 152 QA pairs |
| B (mixed) | LoCoMo + LongMemEval | 212 QA pairs |
| C (LongMemEval-only) | LongMemEval only | 60 QA pairs |

All evaluated on: LoCoMo test (1307 Qs), LongMemEval test (415 Qs).

## Dataset Question Type Distributions

The two benchmarks test complementary memory skills. LoCoMo is skewed toward
open-domain recall; LongMemEval is balanced across temporal, multi-session,
and knowledge-update reasoning.

### LoCoMo

| Question Type | Train (152) | Test (1307) |
|---------------|-------------|-------------|
| open_domain | 87 (57.2%) | 705 (53.9%) |
| multi_hop | 29 (19.1%) | 279 (21.3%) |
| single_hop | 28 (18.4%) | 238 (18.2%) |
| temporal | 8 (5.3%) | 85 (6.5%) |

### LongMemEval

| Question Type | Train (60) | Test (415) |
|---------------|------------|------------|
| temporal-reasoning | 10 (16.7%) | 119 (28.7%) |
| multi-session | 10 (16.7%) | 118 (28.4%) |
| knowledge-update | 10 (16.7%) | 64 (15.4%) |
| single-session-user | 10 (16.7%) | 55 (13.3%) |
| single-session-assistant | 10 (16.7%) | 39 (9.4%) |
| single-session-preference | 10 (16.7%) | 20 (4.8%) |

LongMemEval's train split is perfectly balanced (10 per type by design).
LoCoMo's train split mirrors its test distribution, dominated by open_domain.
This asymmetry is part of what makes the curriculum comparison interesting —
Config B (mixed) exposes the model to question types that Config A never sees
during training.

## Architecture

Our implementation follows the Memory-R1 architecture (Yan et al., 2025).
"Memory" here refers to an external structured store that the agent maintains
across conversation sessions — not the LLM's internal weights or context window.

- Memory Manager: RL-trained agent that decides CRUD operations (ADD/UPDATE/DELETE/NOOP) per dialogue turn
- Answer Agent: RL-trained agent that retrieves relevant memories and generates answers
- Memory Bank: external key-value store with provenance (source session, timestamps)
- Base model: Qwen-2.5-7B-Instruct
- RL algorithm: GRPO (Group Relative Policy Optimization, no critic network)
- Retrieval: embedding-based RAG (sentence-transformers, top-60 candidates)

## Project Structure

```
rl-memory-curriculum/
├── configs/                    # Training and eval YAML configs
│   ├── train_locomo_only.yaml       # Config A (LoRA, default)
│   ├── train_mixed.yaml             # Config B (LoRA, default)
│   ├── train_longmemeval_only.yaml  # Config C (LoRA, default)
│   ├── eval.yaml                    # Evaluation config
│   └── full_ft/                     # Full FT variants (≥80GB GPU)
├── data/                       # Data preparation + processed JSONL
├── src/                        # Core implementation
│   ├── train_grpo.py                # GRPO training (TRL GRPOTrainer)
│   ├── reward.py                    # Reward functions (EM + format)
│   ├── memory_bank.py               # External memory store
│   ├── memory_manager.py            # CRUD policy (RL-trained)
│   ├── answer_agent.py              # Memory-augmented QA (RL-trained)
│   ├── retriever.py                 # Embedding-based retrieval
│   └── pipeline.py                  # End-to-end inference
├── eval/                       # Evaluation, judge, analysis
├── scripts/                    # setup.sh, run_all.sh
├── results/                    # Phase 1 reference results
└── paper/                      # Figures and tables
```

## Metrics

- F1 (token-level, primary — deterministic, compatible with Memory-R1's evaluation)
- BLEU-1 (lexical overlap)
- Exact Match (binary, used as RL reward signal)
- LLM-as-Judge (OpenAI-compatible API, secondary)

## Running Phase 2 on H100

Everything is ready to run. No code changes needed — just pick a config tier.

### Full FT on H100 80GB (recommended)

```bash
git clone <repo> && cd rl-memory-curriculum
bash scripts/setup.sh
bash scripts/run_all.sh --config-dir configs/full_ft
```

This trains 3 configs (AA + MM each), evaluates all 7 models on both benchmarks,
and generates paper tables. ~14h, ~$40-80 on Lambda Labs.

The LoRA path also works on H100 (just omit `--config-dir`). Both tiers produce
checkpoints in the same `checkpoints/` structure, and eval handles both transparently.

### What the pipeline does

1. Data prep (processes raw data if `data/processed/` is empty)
2. Trains Answer Agent for configs A, B, C (skips if checkpoint exists)
3. Trains Memory Manager for configs A, B, C (skips if checkpoint exists)
4. Evaluates all 7 models (baseline + 3 AA-only + 3 AA+MM) on 2 benchmarks
5. Generates paper tables in `paper/tables/`

### Optional: LLM-as-Judge

The pipeline runs `--skip-judge` by default. To add judge scores after eval:

```bash
export OPENAI_API_KEY=sk-...
uv run python eval/run_eval.py --config configs/eval.yaml --judge-only
```

Any OpenAI-compatible endpoint works — set `OPENAI_BASE_URL` for local/proxy APIs.

### Optional tuning (not required)

- `configs/full_ft/*.yaml`: `batch_size` is 2 by default, can go to 4 on H100 for faster training
- `configs/eval.yaml`: `batch_size` is 8, can go higher on H100 for faster eval
- Checkpoint resume is built in — if training crashes, re-run the same command and it skips completed configs

### Phase 1 reference results

`results/` contains Phase 1 results (Qwen-2.5-3B + LoRA + AA-only). See
`results/README.md` for details. Phase 2 training will overwrite these with
7B results.

## Acknowledgments

This project implements and extends the architecture described in
[Memory-R1](https://arxiv.org/abs/2508.19828) (Yan et al., 2025). Memory-R1
has no public code — our implementation is based on the paper's description.
The training data comes from [LoCoMo](https://arxiv.org/abs/2402.17753) and
[LongMemEval](https://arxiv.org/abs/2410.10813).

## References

- [Memory-R1](https://arxiv.org/abs/2508.19828) — Yan et al., 2025 (architecture reference)
- [LoCoMo](https://arxiv.org/abs/2402.17753) — Maharana et al., ACL 2024 (benchmark)
- [LongMemEval](https://arxiv.org/abs/2410.10813) — Wu et al., ICLR 2025 (benchmark)
- [GRPO](https://arxiv.org/abs/2402.03300) — Shao et al., 2024 (RL algorithm)

## License

MIT
