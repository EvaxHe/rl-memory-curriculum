"""
Training callbacks for GRPO training.
"""
import json
import logging
from pathlib import Path
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class RewardLoggingCallback(TrainerCallback):
    """Print reward statistics to stdout for real-time monitoring."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        reward_keys = [k for k in logs if "reward" in k.lower()]
        if reward_keys:
            stats = {k: f"{logs[k]:.4f}" for k in reward_keys}
            logger.info(f"Step {state.global_step} rewards: {stats}")


class TrainingLogCallback(TrainerCallback):
    """Write structured JSONL training logs for paper figures.

    Captures per-step: loss, reward stats, grad norm, learning rate, wall time.
    For MM training, also captures CRUD operation distribution from completions.

    Output: one JSONL file per training run at `logs/{run_name}_training.jsonl`.
    Each line is a self-contained JSON object — crash-safe, easy to parse with pandas.
    """

    def __init__(self, log_dir="logs", agent_type="aa"):
        self.log_dir = Path(log_dir)
        self.agent_type = agent_type
        self._file = None
        self._start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        import time
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / f"{args.run_name}_training.jsonl"
        self._file = open(log_path, "a", encoding="utf-8")
        self._start_time = time.time()
        logger.info(f"Training log: {log_path}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        import time
        if logs is None or self._file is None:
            return

        record = {
            "step": state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch else 0,
            "wall_time_s": round(time.time() - self._start_time, 1),
            "agent": self.agent_type,
        }

        # Standard training metrics
        for key in ("loss", "grad_norm", "learning_rate"):
            if key in logs:
                record[key] = round(logs[key], 6)

        # Reward metrics (TRL logs these with various key patterns)
        for key, val in logs.items():
            if "reward" in key.lower():
                record[key.replace("/", "_")] = round(val, 6)

        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._file is not None:
            self._file.close()
            self._file = None


class RewardVarianceEarlyStopCallback(TrainerCallback):
    """Stop training when reward variance collapses (all GRPO samples score the same).

    When std(rewards) < threshold for `patience` consecutive logging steps,
    GRPO advantages are effectively zero and further training is wasted compute.
    """

    def __init__(self, std_threshold=0.01, patience=20):
        self.std_threshold = std_threshold
        self.patience = patience
        self.low_variance_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # TRL >= 0.29 logs reward/std; some versions use rewards/std
        reward_std = None
        for key in ("reward/std", "rewards/std", "reward_std"):
            if key in logs:
                reward_std = logs[key]
                break
        if reward_std is not None and reward_std < self.std_threshold:
            self.low_variance_count += 1
            if self.low_variance_count >= self.patience:
                logger.warning(
                    f"Reward variance collapsed (std={reward_std:.4f} for "
                    f"{self.patience} consecutive steps). Stopping early."
                )
                control.should_training_stop = True
        else:
            self.low_variance_count = 0
