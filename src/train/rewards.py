"""
Reward functions for GRPO training.

These are the reward functions passed to GRPOTrainer — they have the
specific signature (completions, answer=None, **kwargs) -> list[float]
that TRL expects.
"""
import json
import logging
import re

from src.common.scoring import normalize_answer, token_f1, exact_match
from src.agents.answer_agent import extract_answer_from_completion

logger = logging.getLogger(__name__)


def make_aa_reward_func(reward_type: str = "f1"):
    """
    Factory: create the AA reward function based on config.
    reward_type: "f1" (continuous) or "em" (binary, matches paper).
    """
    score_fn = token_f1 if reward_type == "f1" else exact_match

    def reward_func(completions, answer, **kwargs) -> list[float]:
        rewards = []
        for completion, gold in zip(completions, answer):
            response = completion[0]["content"] if isinstance(completion, list) else str(completion)
            extracted = extract_answer_from_completion(response)
            rewards.append(score_fn(extracted, gold))
        return rewards

    return reward_func


def format_reward_func(completions, **kwargs) -> list[float]:
    """
    Reward for proper XML format output.
    Kept small (0.2 max) so task reward dominates.
    """
    rewards = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score = 0.0
        if "<answer>" in response and "</answer>" in response:
            score += 0.1
        if "<reasoning>" in response and "</reasoning>" in response:
            score += 0.05
        if "<selected_memories>" in response and "</selected_memories>" in response:
            score += 0.05
        rewards.append(score)
    return rewards


def mm_format_reward(completions, **kwargs) -> list[float]:
    """MM reward: valid JSON format + correct CRUD structure."""
    rewards = []
    for completion in completions:
        response = completion[0]["content"] if isinstance(completion, list) else str(completion)
        score = 0.0
        try:
            match = re.search(r"\{[^}]+\}", response)
            if match:
                parsed = json.loads(match.group())
                op = parsed.get("op", "").upper()
                if op in ("ADD", "UPDATE", "DELETE", "NOOP"):
                    score += 0.5
                if op == "ADD" and parsed.get("content", ""):
                    score += 0.3
                elif op == "UPDATE" and parsed.get("entry_id") and parsed.get("content"):
                    score += 0.3
                elif op == "DELETE" and parsed.get("entry_id"):
                    score += 0.3
                elif op == "NOOP":
                    score += 0.2
        except (json.JSONDecodeError, AttributeError):
            pass
        rewards.append(score)
    return rewards


def make_mm_quality_reward():
    """
    Factory: create the MM quality reward function.

    Measures whether the MM's CRUD action *improved* the memory bank's
    relevance to the gold answer. This is the differentiating signal GRPO
    needs — format reward alone saturates (all samples score ~0.8),
    collapsing advantages to zero.

    How it works:
      1. Embed the gold answer (the question this conversation will be asked)
      2. Compute similarity between gold answer and the current memory bank
         (the "before" state, from the prompt's ## Current Memories section)
      3. Simulate the MM's action: if ADD, append content to bank; if NOOP, no change
      4. Compute similarity between gold answer and the updated bank ("after" state)
      5. Reward = max(0, after_sim - before_sim)  (positive delta = improvement)
    """
    # Pre-load embedding model once (lazy, cached globally in retriever.py)
    from src.memory.retriever import embed_texts as _embed_texts

    def mm_quality_reward(completions, answer, **kwargs) -> list[float]:
        rewards = []
        for completion, gold in zip(completions, answer):
            response = completion[0]["content"] if isinstance(completion, list) else str(completion)
            score = 0.0
            try:
                if not gold or not gold.strip():
                    rewards.append(0.0)
                    continue

                match = re.search(r"\{[^}]+\}", response)
                if not match:
                    rewards.append(0.0)
                    continue

                parsed = json.loads(match.group())
                op = parsed.get("op", "").upper()
                content = parsed.get("content", "")

                # Embed the gold answer
                gold_emb = _embed_texts([gold])
                if gold_emb is None:
                    rewards.append(0.0)
                    continue

                # Extract current memories from the prompt context
                prompt_text = ""
                if "prompts" in kwargs:
                    prompts = kwargs["prompts"]
                    idx = len(rewards)
                    if idx < len(prompts):
                        p = prompts[idx]
                        prompt_text = p[-1]["content"] if isinstance(p, list) else str(p)

                # Parse existing memories from prompt
                existing_mems = []
                for line in prompt_text.split("\n"):
                    line = line.strip()
                    if line.startswith("- [") and "] " in line:
                        mem_content = line.split("] ", 1)[1] if "] " in line else line
                        existing_mems.append(mem_content)

                # Compute "before" similarity
                before_sim = 0.0
                if existing_mems:
                    bank_emb = _embed_texts(existing_mems)
                    if bank_emb is not None:
                        sims = bank_emb @ gold_emb.T
                        before_sim = float(sims.max())

                # Simulate the action and compute "after" similarity
                after_mems = list(existing_mems)
                if op == "ADD" and content:
                    after_mems.append(content)
                elif op == "DELETE" and parsed.get("entry_id") is not None:
                    try:
                        del_idx = int(parsed["entry_id"])
                        if 0 <= del_idx < len(after_mems):
                            after_mems.pop(del_idx)
                    except (ValueError, IndexError):
                        pass
                elif op == "UPDATE" and parsed.get("entry_id") is not None and content:
                    try:
                        upd_idx = int(parsed["entry_id"])
                        if 0 <= upd_idx < len(after_mems):
                            after_mems[upd_idx] = content
                    except (ValueError, IndexError):
                        pass

                after_sim = 0.0
                if after_mems:
                    after_emb = _embed_texts(after_mems)
                    if after_emb is not None:
                        sims = after_emb @ gold_emb.T
                        after_sim = float(sims.max())

                # Reward = positive delta (improvement), clamped to [0, 1]
                delta = after_sim - before_sim
                score = max(0.0, min(delta, 1.0))

            except (json.JSONDecodeError, AttributeError, Exception) as e:
                logger.debug(f"MM quality reward error: {e}")
                score = 0.0

            rewards.append(score)
        return rewards

    return mm_quality_reward
