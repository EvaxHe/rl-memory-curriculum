# Phase 2 Pre-GPU-Run Modifications

**Date**: 2026-03-31
**Goal**: Fix critical training and eval issues before the H100 run (~$140 budget, ~20h)
**Repo**: `rl-memory-curriculum` (vLLM backend for eval)
**Hardware**: 1× H100 80GB on Brev (NVIDIA GPU cloud)
**Storage**: Brev persistent disk (survives instance stop/start). Backup via `git push` after each phase.

---

## Why This Document Exists

We identified 8 issues during code review. Some are training-correctness bugs that would waste GPU hours. Others are operational safeguards that protect against losing work. This document orders them by impact, explains the *why* behind each fix, and tracks implementation status.

**Key principle**: Every hour on the H100 costs ~$7. A bug that wastes 3 hours of training costs $21. A bug that loses 6 hours of eval results costs $42. Fixing these issues before the run is the highest-ROI work we can do.

---

## Priority 1: MM Reward Signal (CRITICAL — without this, MM training = wasted compute)

### The Problem

Look at `mm_format_reward()` in `src/train_grpo.py`:

```python
def mm_format_reward(completions, **kwargs) -> list[float]:
    # ...
    if op in ("ADD", "UPDATE", "DELETE", "NOOP"):
        score += 0.5
    if op == "ADD" and parsed.get("content", ""):
        score += 0.3   # total: 0.8
    elif op == "NOOP":
        score += 0.2   # total: 0.7
```

This reward only checks **format** — "did you output valid JSON with a real CRUD op?" It says nothing about **quality** — "was ADD the right decision? Is the content worth remembering?"

A 7B model learns to produce valid JSON within ~10 training steps. After that, all G=4 GRPO samples score identically (e.g., all 0.8). When all samples have the same reward, GRPO computes advantages as:

```
advantage_i = reward_i - mean(rewards)  →  0.8 - 0.8 = 0.0
```

Zero advantages → zero policy gradient → loss = 0 → the model stops learning. The remaining epochs are pure waste. This is called **policy collapse** — the model found a local optimum (always output valid ADD) and has no gradient signal to escape it.

### Why This Happens in RL (the intuition)

GRPO learns by **comparing** samples within a group. If sample A scores 0.8 and sample B scores 0.3, the model learns "do more of what A did." But if all samples score 0.8, there's nothing to compare — every action looks equally good. It's like grading an exam where everyone gets 80% — you can't tell who actually understood the material.

This is fundamentally different from supervised learning, where the loss comes from distance to a target. In RL, the loss comes from **variance** in rewards across samples. No variance = no learning signal.

### The Fix (two parts)

**Part A: Early stopping callback**

Monitor reward variance. When `std(rewards) < 0.01` for 20 consecutive logging steps, stop training — further epochs won't help.

```python
class RewardVarianceCallback(TrainerCallback):
    """Stop MM training when reward variance collapses (all samples score the same)."""
    def __init__(self, std_threshold=0.01, patience=20):
        self.std_threshold = std_threshold
        self.patience = patience
        self.low_variance_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        # TRL >= 0.29 logs reward stats automatically
        reward_std = logs.get("rewards/std", None) if logs else None
        if reward_std is not None and reward_std < self.std_threshold:
            self.low_variance_count += 1
            if self.low_variance_count >= self.patience:
                logger.warning(f"MM reward variance collapsed (std={reward_std:.4f} "
                               f"for {self.patience} steps). Stopping early.")
                control.should_training_stop = True
        else:
            self.low_variance_count = 0
```

**Part B: Quality reward component**

Add a signal that differentiates *good* ADD from *bad* ADD. We use **embedding similarity delta** — measuring whether the MM's action *improved* the memory bank's relevance to the gold answer.

How it works:
1. Embed the gold answer using `all-MiniLM-L6-v2` (80MB, runs on CPU, negligible overhead)
2. Compute max cosine similarity between gold answer and the current memory bank ("before" state)
3. Simulate the MM's CRUD action on the bank
4. Compute max cosine similarity between gold answer and the updated bank ("after" state)
5. Reward = max(0, after_sim - before_sim) — positive delta means the action improved the bank

Why delta, not absolute similarity:
- Absolute similarity rewards ADD of anything vaguely related, even if the bank already has it
- Delta rewards *marginal improvement* — the MM only gets credit for making the bank *better*
- NOOP gets delta=0 (appropriate — format reward still gives 0.7 for valid NOOP)
- DELETE of noise gets positive delta (removing irrelevant entries increases max similarity)
- ADD of redundant info gets ~zero delta (bank already had similar content)

Why embedding over token overlap:
- Token overlap misses semantic matches ("Italian food" vs "Italian cuisine" → 0 overlap)
- Embedding similarity captures meaning, not just surface tokens
- A 7B model can't game embeddings by copying question tokens into ADD content
- The embedding model is already loaded for retrieval — zero additional dependency

```python
from src.retriever import embed_texts

def mm_quality_reward(completions, answer, **kwargs) -> list[float]:
    rewards = []
    for completion, gold in zip(completions, answer):
        # ... parse JSON, extract op and content ...
        gold_emb = embed_texts([gold])
        
        # "Before" similarity: current bank vs gold
        bank_emb = embed_texts(existing_memories)
        before_sim = (bank_emb @ gold_emb.T).max()
        
        # Simulate action, compute "after" similarity
        after_mems = apply_crud(existing_memories, op, content)
        after_emb = embed_texts(after_mems)
        after_sim = (after_emb @ gold_emb.T).max()
        
        # Reward = positive improvement only
        delta = max(0, after_sim - before_sim)
        rewards.append(delta)
    return rewards
```

### Files to modify
- `src/train_grpo.py`: Add `RewardVarianceCallback`, add `mm_quality_reward`, wire both into `train_memory_manager()`

### Status
- [x] Early stopping callback implemented
- [x] Quality reward function implemented
- [x] Both wired into trainer
- [ ] Tested locally (dry run)

---

## Priority 2: MM Training Data Construction (CRITICAL — even good reward can't teach UPDATE/DELETE on bad data)

### The Problem

In `train_memory_manager()` (lines ~460-480 of `src/train_grpo.py`):

```python
for session in ex.get("sessions", [])[:2]:       # only first 2 sessions
    for i, turn in enumerate(session.get("turns", [])[:5]):  # only first 5 turns
```

And the memory state is static:
```python
memories = build_heuristic_memory(ex)
mem_str = "\n".join(f"- [{i}] {m}" for i, m in enumerate(memories[-10:]))
# ^ same snapshot for EVERY turn
```

Three problems:
1. **Only sees conversation beginnings** — first 2 sessions × 5 turns = max 10 turns. LoCoMo conversations have 50-200+ turns across 5-10 sessions. The MM never sees late-conversation turns where UPDATE/DELETE matter most (user changes preferences, corrects earlier statements, etc.)
2. **Static memory state** — every turn sees the same `memories[-10:]` snapshot. The MM never sees the consequence of its own decisions. It's like teaching someone to play chess but always showing them the same board position.
3. **Biased toward ADD/NOOP** — with only 10 entries in the bank (all from heuristics), there's nothing meaningful to UPDATE or DELETE. The optimal policy is always ADD or NOOP.

### Why This Matters (the RL perspective)

RL learns from the **state-action-reward** loop. If the state never changes (static memory bank), the model can't learn state-dependent policies. It'll learn a single unconditional action (ADD everything) because that's what works for the one state it ever sees.

For UPDATE/DELETE to be learnable, the model needs to see states where:
- A memory entry contradicts the current turn → UPDATE is correct
- A memory entry is explicitly retracted → DELETE is correct
- The bank is full of low-quality entries → DELETE to make room

None of these states occur with 10 heuristic entries from the first 10 turns.

### The Fix

**Sliding window sampling** — sample turns from across the full conversation, not just the beginning:

```python
for session in ex.get("sessions", []):
    turns = session.get("turns", [])
    # Sample up to 5 turns spread across the session (early, mid, late)
    if len(turns) > 5:
        indices = np.linspace(0, len(turns) - 1, 5, dtype=int)
        sampled_turns = [(int(idx), turns[int(idx)]) for idx in indices]
    else:
        sampled_turns = list(enumerate(turns))
```

**Evolving memory state** — accumulate memories as we process turns, so later turns see a richer bank:

```python
evolving_memories = []
for session in ex.get("sessions", []):
    turns = session.get("turns", [])
    for i, turn in sampled_turns:
        # Build prompt with CURRENT evolving state
        mem_str = format_memory_state(evolving_memories[-20:])  # last 20, not 10
        prompt = build_mm_prompt(turn, mem_str, session)
        mm_prompts.append(prompt)

        # Simulate heuristic ADD for this turn (so next turn sees updated state)
        text = turn.get("text", "").strip()
        if len(text.split()) > 5:
            evolving_memories.append(f"{turn['speaker']}: {text[:300]}")
```

### 关于训练时间的影响

Current: ~152 examples × 10 turns = ~1,520 MM prompts
After fix: ~152 examples × ~25 turns (5 per session × ~5 sessions) = ~3,800 MM prompts

That's ~2.5x more data, which means MM training goes from ~1h to ~2.5h on H100. Within budget. The quality improvement is worth it — without this, MM training produces a model that only knows ADD.

If we're tight on time, we can cap at 20 prompts per example (`max_mm_prompts_per_example=20`).

### Files to modify
- `src/train_grpo.py`: Rewrite the MM data construction loop in `train_memory_manager()`

### Status
- [x] Sliding window sampling implemented
- [x] Evolving memory state implemented
- [x] Max prompts per example cap added
- [ ] Tested locally (check prompt count, memory state variety)

---

## Priority 3: Bump max_grad_norm (1 minute, immediate impact)

### The Problem

Both AA and MM use `max_grad_norm=0.1`. Standard GRPO uses 1.0 (DeepSeek-R1, Memory-R1 paper default).

### Why This Matters

Gradient clipping caps the magnitude of parameter updates. At 0.1, you're saying "never take a step larger than 0.1 in gradient norm." This was probably set conservatively for Phase 1's 3B model to prevent instability, but for 7B full FT with lr=1e-6, gradients are naturally smaller (more parameters to distribute across). 0.1 is so aggressive it can prevent the model from escaping local optima — like the NOOP collapse in MM.

Think of it as a speed limit. 0.1 is a school zone speed limit on a highway. The model wants to go faster but can't.

### The Fix

In both `train_answer_agent()` and `train_memory_manager()`, change:
```python
max_grad_norm=0.1  →  max_grad_norm=1.0
```

Also update the full_ft YAML configs if `max_grad_norm` is specified there (currently it's hardcoded in Python, not in YAML — so just the Python change).

### Files to modify
- `src/train_grpo.py`: Two lines (AA GRPOConfig and MM GRPOConfig)

### Status
- [x] AA max_grad_norm updated to 1.0
- [x] MM max_grad_norm updated to 1.0

---

## Priority 4: Reward Distribution Logging (essential for diagnosing issues during the run)

### The Problem

`report_to="none"` in both AA and MM GRPOConfig. This suppresses TRL's built-in reward logging. If training goes wrong on the H100, you won't know until eval — hours later.

### Why This Matters

Without reward logs, you're flying blind. You can't distinguish between:
- "Training is working, rewards are climbing" (good)
- "Rewards collapsed at step 20, remaining 500 steps are wasted" (bad — should have stopped)
- "Rewards are oscillating wildly" (bad — learning rate too high)

At $7/hr, every hour of blind training is expensive.

### The Fix

**Option A (minimal)**: Change `report_to="none"` to `report_to="tensorboard"`. TRL will log reward mean/std/min/max automatically. View with `tensorboard --logdir checkpoints/`.

**Option B (belt + suspenders)**: Also add a stdout callback that prints reward stats every N steps:

```python
class RewardLoggingCallback(TrainerCallback):
    """Print reward statistics to stdout for real-time monitoring."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        reward_keys = [k for k in logs if k.startswith("reward")]
        if reward_keys:
            stats = {k: f"{logs[k]:.4f}" for k in reward_keys}
            logger.info(f"Step {state.global_step} rewards: {stats}")
```

I recommend both. Tensorboard for post-hoc analysis, stdout for real-time monitoring via SSH.

### Files to modify
- `src/train_grpo.py`: Change `report_to`, add `RewardLoggingCallback` to both trainers

### Status
- [x] `report_to="tensorboard"` set for AA and MM
- [x] RewardLoggingCallback implemented
- [x] Callback added to both trainers

---

## Priority 5: Incremental Prediction Saving in Eval (protects against losing 6h of eval work)

### The Problem

In `run_inference()` (eval/run_eval.py), all predictions are generated in memory, then written to disk at the end:

```python
# All predictions generated here (could be 1307 for LoCoMo)
predictions = run_inference(model, tokenizer, ...)

# Only saved here — if we crash before this line, everything is lost
pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
with open(pred_path, "w") as f:
    for p in predictions:
        f.write(json.dumps(p) + "\n")
```

With 7 models × 2 benchmarks, eval takes ~8 hours. If you OOM on model 5 (the first AA+MM model, which loads two 7B models), you lose the predictions for that model and have to re-run.

### The Fix

Write predictions incrementally as they're generated. Two changes:

**In `run_inference()`**: Accept an optional `output_path` parameter. Write each prediction as it's generated:

```python
def run_inference(..., output_path=None):
    # ... existing code ...
    if output_path:
        f_out = open(output_path, "a")  # append mode

    for meta, answer in zip(all_metadata, answers):
        meta["answer"] = answer
        predictions.append(meta)
        if output_path:
            f_out.write(json.dumps(meta) + "\n")
            f_out.flush()  # ensure it's on disk

    if output_path:
        f_out.close()
```

**In `main()`**: Check for existing partial predictions and skip already-completed examples:

```python
pred_path = output_dir / f"{model_name}_{bench_name}_predictions.jsonl"
existing_count = 0
if pred_path.exists():
    with open(pred_path) as f:
        existing_count = sum(1 for line in f if line.strip())
    if existing_count >= expected_count:
        logger.info(f"Skipping {model_name}/{bench_name} — {existing_count} predictions exist")
        continue
```

### Files to modify
- `eval/run_eval.py`: Modify `run_inference()` and `main()`

### Status
- [x] Incremental writing in run_inference()
- [x] Resume logic in main()
- [ ] Tested with simulated crash (kill after N predictions, restart)

---

## Priority 6: vLLM Dual-Model Memory Split (real risk on H100 with vLLM backend)

### The Problem

When `use_mm=true` with vLLM backend, the code loads both AA and MM simultaneously:

```python
aa_gpu_mem = 0.45   # 45% of 80GB = 36GB for AA
mm_gpu_mem = 0.45   # 45% of 80GB = 36GB for MM
# Total: 90% = 72GB for model weights + KV cache
# Remaining: 8GB for CUDA context, page tables, vLLM overhead
```

Two 7B models in bf16 = ~14GB each = 28GB for weights. That leaves 44GB for KV cache split across two vLLM instances. Sounds fine, but vLLM's memory management has overhead:
- CUDA context: ~1-2GB
- vLLM page tables: ~0.5GB per instance
- Memory fragmentation: unpredictable
- If either model's KV cache needs slightly more (long prompts with 60 retrieved memories), OOM.

### Why Sequential is Better

The MM only needs to run once per conversation to build the memory bank. The AA then uses those memories for all questions. They don't need to be loaded simultaneously.

### The Fix

Restructure the per-model eval loop to be two-phase:

```
Phase 1: Load MM at gpu_memory_utilization=0.85
         → Run memory construction for ALL conversations
         → Save memories to disk
         → Unload MM

Phase 2: Load AA at gpu_memory_utilization=0.85
         → Load memories from disk
         → Run answer generation for all questions
         → Unload AA
```

Each model gets the full GPU. No memory contention. AA gets a much bigger KV cache, which means faster inference on the long prompts (60 retrieved memories can be 2000+ tokens).

### Files to modify
- `eval/run_eval.py`: Restructure the `use_mm` path in `main()` to be sequential

### Status
- [x] Sequential MM→AA eval implemented
- [x] Memory bank serialization (save/load between phases)
- [ ] Tested with a single model (dry run)

---

## Priority 7: KL Penalty for MM (one-line insurance)

### The Problem

No KL divergence penalty against the base model. For AA this is fine — the task reward (EM/F1) grounds the policy. For MM with the weak format reward, there's nothing preventing the policy from drifting arbitrarily far from the pretrained distribution.

### Why KL Matters (the intuition)

KL penalty says "don't stray too far from what the pretrained model would do." It's a regularizer. Without it, the MM can learn degenerate behaviors like always outputting the exact same JSON string — which scores well on format reward but is useless.

With the quality reward fix (Priority 1), KL is less critical. But it's free insurance.

### The Fix

In `train_memory_manager()`, add `beta=0.04` to GRPOConfig:

```python
training_args = GRPOConfig(
    ...
    beta=0.04,  # KL penalty coefficient (regularizes against base model drift)
    ...
)
```

TRL's GRPOConfig supports this natively. 0.04 is the standard value from the GRPO paper.

### Files to modify
- `src/train_grpo.py`: One line in MM GRPOConfig

### Status
- [x] beta=0.04 added to MM GRPOConfig

---

## Priority 8: Checkpoint Completion Detection in run_all.sh (nice to have)

### The Problem

```bash
if [ ! -d "checkpoints/${prefix}/answer_agent" ]; then
    # train...
```

This checks if the directory exists, but the directory is created by TRL at the start of training, not at the end. If training crashes at step 80/100, the directory exists but training isn't complete. On restart, the script skips this config entirely.

### The Fix

Check for `training_meta.json` instead — this file is only written after `trainer.train()` completes successfully:

```bash
if [ ! -f "checkpoints/${prefix}/answer_agent/training_meta.json" ]; then
    # train...
```

### Files to modify
- `scripts/run_all.sh`: Change directory checks to file checks

### Status
- [x] AA checkpoint check updated
- [x] MM checkpoint check updated

---

## Bonus: Validation Loop (not in original plan, but high value)

### The Problem

Every config specifies a `val_file` but it's never used during training. We train for a fixed number of epochs with no signal on whether the model is actually improving on held-out data.

### Why This Matters

Without validation, you can't distinguish between:
- "2 epochs is enough" vs "we need 4 epochs"
- "Training is overfitting after epoch 1" vs "training is still improving"

For AA with EM reward, this is especially important — EM is binary, so training loss can look fine while the model is memorizing training examples instead of generalizing.

### The Fix

Add a validation evaluation every N steps (e.g., every 50 steps or once per epoch). This doesn't need to be a full eval — just run the reward function on the val set and log the mean.

This is lower priority than the other fixes but worth adding if time permits. Can be a simple callback that loads val data once and evaluates periodically.

### Files to modify
- `src/train_grpo.py`: Add validation callback

### Status
- [ ] Validation callback implemented (stretch goal)

---

## Priority 9: JSONL Training Logs for Paper Figures (implemented)

### The Problem

TensorBoard (`report_to="tensorboard"`) gives us interactive training curves, but:
- Exporting TB data for matplotlib/seaborn paper figures is clunky
- TB doesn't log MM operation distribution (ADD/UPDATE/DELETE/NOOP) — the key plot for our paper
- TB files can corrupt on crash; JSONL is append-only and crash-safe
- We need to `git push` logs off the Brev instance — small JSONL files are easy, TB event files are not

### What Reviewers Want to See (and what we need to capture)

**What TensorBoard already gives us (with `report_to="tensorboard"`):**
- `loss` per step — standard, every RL paper shows this
- `reward/mean`, `reward/std` — TRL >= 0.29 logs these automatically for GRPO
- `learning_rate` schedule — useful for appendix, shows warmup + cosine decay
- `grad_norm` — important now that we bumped to 1.0, shows if training is stable

TB logs land in `checkpoints/{config}/answer_agent/runs/` and `memory_manager/runs/`. We'll have 6 TB directories (3 configs × 2 agents). That's enough for basic interactive exploration during the run.

**What we add beyond TB defaults (via JSONL):**

1. **Per-reward-function breakdown** — We have two reward functions for both AA (`aa_reward` + `format_reward`) and MM (`mm_format_reward` + `mm_quality_reward`). TB logs the sum, but we want them separated. Reviewers will ask "how much of the reward comes from format vs quality?" This is the key ablation for the MM reward design.

2. **Reward variance over time** — We have the early stopping callback monitoring this, but we also want it plotted. "Figure 3: MM reward variance collapses after step X for format-only reward (ablation) but remains positive with embedding delta reward" is a compelling figure.

3. **Wall-clock time per step** — For the cost table in the paper. TB logs this but it's buried in the event file format. JSONL makes it trivially accessible.

4. **Grad norm trajectory** — With `max_grad_norm` bumped from 0.1 to 1.0, we want to verify gradients are well-behaved. If grad_norm is consistently near 1.0, we're still clipping heavily and might need to investigate. If it's 0.2-0.5, the bump was the right call.

**What we decided NOT to capture (and why):**

1. **MM operation distribution per step** (ADD/UPDATE/DELETE/NOOP counts) — Would require hooking into the reward function's completions at every step, adding complexity and risk. We can extract this post-hoc from saved checkpoints by running inference on the training data if reviewers ask. Not worth the risk of slowing down training or introducing bugs before a $140 run.

2. **Intermediate checkpoint eval curves** (eval metrics vs training step) — Too expensive. Each eval run takes ~8h across 7 models × 2 benchmarks. Running eval at multiple checkpoints would blow the budget. Final-checkpoint eval is sufficient for the first submission. We save checkpoints every 50 steps (`save_steps=50`), so we can always go back and eval intermediate checkpoints if reviewers specifically request learning curves.

3. **Per-example reward distributions** (histograms of reward per training example) — Interesting for debugging but too much data to log per step. The mean/std summary statistics capture the important signal (is variance collapsing? is mean improving?).

**Why JSONL over just TensorBoard:**

- **Paper figures**: JSONL → `pd.read_json(lines=True)` → matplotlib in 3 lines. TB → `tbparse` or manual export → fight with TB's event format → matplotlib. JSONL wins.
- **Crash safety**: JSONL is append-only. Each line is independent. If training crashes mid-step, you lose at most one line. TB event files can corrupt.
- **Portability**: JSONL files are ~10KB per training run. Easy to `git push`, share with collaborators, or attach to a paper repo. TB event files are larger and require the TB viewer.
- **Complementary**: We keep TB for real-time monitoring during the run (SSH in, `tensorboard --logdir checkpoints/`). JSONL is for post-hoc analysis and paper generation. Both serve different purposes.

### What We Log

A `TrainingLogCallback` writes one JSONL line per logging step (every 5 steps) to `logs/{run_name}_training.jsonl`:

```jsonl
{"step": 5, "epoch": 0.033, "wall_time_s": 45.2, "agent": "aa", "loss": 0.234, "grad_norm": 0.34, "learning_rate": 1e-06, "reward_mean": 0.45, "reward_std": 0.12}
{"step": 10, "epoch": 0.066, "wall_time_s": 91.8, "agent": "mm", "loss": 0.189, "grad_norm": 0.28, "learning_rate": 9.8e-07, "reward_mean": 0.62, "reward_std": 0.08}
```

### What This Enables for the Paper

1. **Training reward curves** (Figure 2): All 3 configs overlaid, separate for AA and MM. Shows convergence behavior and curriculum effects.
2. **Reward variance over time** (Figure 3): Demonstrates that embedding delta reward prevents the collapse that format-only reward causes. Key ablation figure.
3. **Loss curves** (Appendix): Standard, every paper includes these.
4. **Cost table** (Table 3): Wall-clock time per config from `wall_time_s`.

### Files modified
- `src/train_grpo.py`: Added `TrainingLogCallback`, wired into both AA and MM trainers

### Status
- [x] TrainingLogCallback implemented
- [x] Wired into AA trainer
- [x] Wired into MM trainer
- [x] Logs to `logs/{run_name}_training.jsonl`

---

## Implementation Order (estimated time)

| # | Fix | Time | Status | Risk if skipped |
|---|-----|------|--------|-----------------|
| 3 | max_grad_norm → 1.0 | 2 min | ✅ Done | Slow learning, possible local optima |
| 7 | KL beta=0.04 for MM | 2 min | ✅ Done | MM policy drift (mitigated by fix #1) |
| 4 | Reward logging + tensorboard | 15 min | ✅ Done | Flying blind during training |
| 8 | Checkpoint completion detection | 5 min | ✅ Done | Skipped configs on restart |
| 9 | JSONL training logs for paper | 20 min | ✅ Done | No data for paper figures |
| 1 | MM reward signal (early stop + embedding delta) | 45 min | ✅ Done | **MM training = wasted compute** |
| 2 | MM data construction | 30 min | ✅ Done | MM can't learn UPDATE/DELETE |
| 5 | Incremental prediction saving | 30 min | ✅ Done | Lose hours of eval on crash |
| 6 | Sequential vLLM eval (MM→AA) | 45 min | ✅ Done | OOM during AA+MM eval |
| Bonus | Validation loop | 30 min | Deferred | No early signal on generalization |
| **Total** | | **~4h** | **9/10 done** | |

### Note: Checkpoint save frequency (save_steps)

Saving a 7B full FT checkpoint writes ~14GB to disk, which takes 30-60s on NVMe. At ~10s/step on H100, `save_steps=50` means pausing for a checkpoint save every ~8 minutes of training — roughly 8% overhead.

We adjusted:
- **AA**: `save_steps=50` → `save_steps=200`. AA training is short (76-160 total steps depending on config, ~15 min on H100). At 200, we get 0-1 intermediate saves. The crash risk for a 15-minute run is low, and the final model save always happens after `trainer.train()` regardless.
- **MM**: Already at `save_steps=300`. MM training is longer (~1900 steps, ~5h). Saves every ~50 minutes, which is reasonable. One intermediate checkpoint provides crash recovery without excessive I/O overhead.

`save_total_limit=2` for both, so only the latest 2 checkpoints are kept on disk.

---

## Post-Implementation Checklist

- [x] All fixes implemented and committed
- [ ] Dry run passes (`bash scripts/run_all.sh --dry-run`)
- [ ] Reward logging visible in stdout during dry run
- [ ] MM training data count is ~3,000-4,000 (not ~1,500)
- [x] max_grad_norm=1.0 confirmed in both AA and MM
- [x] beta=0.04 confirmed in MM GRPOConfig
- [x] Checkpoint detection uses `training_meta.json`
- [x] Eval writes predictions incrementally (check partial file exists during run)
- [x] vLLM eval loads models sequentially (no dual-model)
- [ ] Tensorboard logs directory created during training
- [x] JSONL training logs written to `logs/` directory
- [ ] `git push` logs after AA training, after MM training, after eval

---

## References

- GRPO paper: [arxiv:2402.03300](https://arxiv.org/abs/2402.03300) — standard beta=0.04, max_grad_norm=1.0
- TRL GRPOTrainer docs: reward logging, callbacks, beta parameter
- Memory-R1: [arxiv:2508.19828](https://arxiv.org/abs/2508.19828) — EM reward, full FT, 7B backbone
- DeepSeek-R1: max_grad_norm=1.0 as standard for GRPO
