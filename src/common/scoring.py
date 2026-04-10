"""
Reward/scoring functions for GRPO training and evaluation.

Memory-R1 uses token-level F1 as the primary reward signal.
We also support BLEU-1 and Exact Match for analysis.
"""
import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answer text for fair comparison."""
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def token_f1(prediction: str, reference: str) -> float:
    """
    Compute token-level F1 between prediction and reference.
    This is the primary reward signal for GRPO training.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def bleu1(prediction: str, reference: str) -> float:
    """Compute BLEU-1 (unigram precision with brevity penalty)."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    clipped = 0
    for token in pred_tokens:
        if ref_counts[token] > 0:
            clipped += 1
            ref_counts[token] -= 1

    precision = clipped / len(pred_tokens)

    # Brevity penalty
    bp = min(1.0, len(pred_tokens) / max(len(ref_tokens), 1))
    return bp * precision


def exact_match(prediction: str, reference: str) -> float:
    """Binary exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def compute_reward(prediction: str, reference: str,
                   reward_type: str = "f1") -> float:
    """Compute reward for GRPO training."""
    if reward_type == "f1":
        return token_f1(prediction, reference)
    elif reward_type == "bleu1":
        return bleu1(prediction, reference)
    elif reward_type == "exact_match":
        return exact_match(prediction, reference)
    elif reward_type == "combined":
        # Weighted combination: 0.7 * F1 + 0.2 * BLEU-1 + 0.1 * EM
        return (0.7 * token_f1(prediction, reference)
                + 0.2 * bleu1(prediction, reference)
                + 0.1 * exact_match(prediction, reference))
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
