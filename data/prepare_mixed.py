"""
Create mixed training set for Config B.

Combines LoCoMo train (152 examples) + LongMemEval train (~60 examples)
into a single shuffled training set of ~212 examples.

The mixed curriculum exposes the RL agent to both:
- Standard conversational memory (LoCoMo: single-hop, multi-hop, open-domain)
- Temporal reasoning + knowledge updates (LongMemEval)
"""
import json
import logging
import random
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/processed")


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} examples to {path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    random.seed(42)

    # Load individual train sets
    locomo_path = OUTPUT_DIR / "locomo_train.jsonl"
    longmemeval_path = OUTPUT_DIR / "longmemeval_train.jsonl"

    if not locomo_path.exists() or not longmemeval_path.exists():
        logger.error("Run prepare_locomo.py and prepare_longmemeval.py first!")
        return

    locomo_train = load_jsonl(locomo_path)
    longmemeval_train = load_jsonl(longmemeval_path)

    logger.info(f"LoCoMo train: {len(locomo_train)} examples")
    logger.info(f"LongMemEval train: {len(longmemeval_train)} examples")

    # Combine and shuffle
    mixed = locomo_train + longmemeval_train
    random.shuffle(mixed)

    logger.info(f"Mixed train: {len(mixed)} examples")

    # Source distribution
    sources = Counter(ex.get("source_benchmark", "unknown") for ex in mixed)
    logger.info(f"Source distribution: {dict(sources)}")

    # Question type distribution
    qtypes = Counter(ex.get("question_type", "unknown") for ex in mixed)
    logger.info(f"Question type distribution: {dict(sorted(qtypes.items()))}")

    # Save
    save_jsonl(mixed, OUTPUT_DIR / "mixed_train.jsonl")

    # Also create a mixed val set
    locomo_val = load_jsonl(OUTPUT_DIR / "locomo_val.jsonl")
    longmemeval_val = load_jsonl(OUTPUT_DIR / "longmemeval_val.jsonl")
    mixed_val = locomo_val + longmemeval_val
    random.shuffle(mixed_val)
    save_jsonl(mixed_val, OUTPUT_DIR / "mixed_val.jsonl")

    logger.info("Mixed data preparation complete!")


if __name__ == "__main__":
    main()
