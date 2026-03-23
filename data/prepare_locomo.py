"""
Prepare LoCoMo benchmark data for Memory-R1 training.

LoCoMo (Maharana et al., ACL 2024): 10 extended conversations with
1986 total QA pairs across 5 categories:
  1=single_hop (282), 2=multi_hop (321), 3=temporal (96),
  4=open_domain (841), 5=adversarial (446)

Memory-R1 excludes adversarial (cat 5) and uses a 152/81/1307 split
on the remaining 1540 QA pairs (from Mem0's protocol).

Raw data format (locomo10.json):
  [{
    "sample_id": "conv-26",
    "qa": [{"question": ..., "answer": ..., "evidence": ["D1:3"], "category": int}],
    "conversation": {
      "speaker_a": "Caroline", "speaker_b": "Melanie",
      "session_1": [{"speaker": "Caroline", "dia_id": "D1:1", "text": "..."}],
      "session_1_date_time": "1:56 pm on 8 May, 2023",
      ...
    },
    "event_summary": ..., "observation": ..., "session_summary": ...
  }]

Output format (JSONL):
  {"conversation_id", "sessions": [{"session_id", "date_time", "turns": [...]}],
   "question", "answer", "question_type", "evidence", "source_benchmark"}
"""
import json
import logging
import random
import subprocess
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

RAW_FILE = Path("data/raw/locomo/data/locomo10.json")  # https://github.com/snap-research/locomo
OUTPUT_DIR = Path("data/processed")

CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}

# Memory-R1 split sizes (from Mem0 protocol, excludes adversarial)
TRAIN_SIZE = 152
VAL_SIZE = 81
# Test = remainder (1540 - 152 - 81 = 1307)


def parse_sessions(conversation: dict) -> list[dict]:
    """Extract sessions from LoCoMo's flat key format into a list."""
    sessions = []
    # Find all session keys (session_1, session_2, ...)
    session_nums = set()
    for key in conversation.keys():
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                num = int(key.split("_")[1])
                session_nums.add(num)
            except (ValueError, IndexError):
                continue

    for num in sorted(session_nums):
        key = f"session_{num}"
        dt_key = f"session_{num}_date_time"
        turns_raw = conversation.get(key, [])
        turns = []
        for t in turns_raw:
            turns.append({
                "speaker": t.get("speaker", ""),
                "dia_id": t.get("dia_id", ""),
                "text": t.get("text", ""),
            })
        sessions.append({
            "session_id": num,
            "date_time": conversation.get(dt_key, ""),
            "turns": turns,
        })

    return sessions


def build_examples(raw_data: list[dict]) -> list[dict]:
    """Convert raw LoCoMo format to our standard JSONL format."""
    examples = []
    for conv_item in raw_data:
        conv_id = conv_item["sample_id"]
        conversation = conv_item["conversation"]
        sessions = parse_sessions(conversation)
        speaker_a = conversation.get("speaker_a", "")
        speaker_b = conversation.get("speaker_b", "")

        for qa in conv_item["qa"]:
            cat_int = qa["category"]
            # Skip adversarial (cat 5) — Memory-R1 excludes these
            if cat_int == 5:
                continue

            examples.append({
                "conversation_id": conv_id,
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
                "sessions": sessions,
                "question": qa["question"],
                "answer": str(qa["answer"]),  # some answers are ints
                "question_type": CATEGORY_NAMES.get(cat_int, f"cat_{cat_int}"),
                "evidence": qa.get("evidence", []),
                "source_benchmark": "locomo",
            })

    return examples


def create_splits(examples: list[dict], seed: int = 42):
    """
    Create train/val/test splits following Memory-R1's protocol.

    Memory-R1 splits at the QA-pair level (not conversation level).
    We shuffle with a fixed seed and take 152 train / 81 val / rest test.
    """
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    train = shuffled[:TRAIN_SIZE]
    val = shuffled[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test = shuffled[TRAIN_SIZE + VAL_SIZE:]

    return train, val, test


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} examples to {path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not RAW_FILE.exists():
        clone_dir = RAW_FILE.parent.parent  # data/raw/locomo
        if not clone_dir.exists():
            logger.info(f"Raw data not found. Cloning LoCoMo repo into {clone_dir}...")
            subprocess.run(
                ["git", "clone", "https://github.com/snap-research/locomo", str(clone_dir)],
                check=True,
            )
        if not RAW_FILE.exists():
            logger.error(f"Raw file still not found after clone: {RAW_FILE}")
            return

    with open(RAW_FILE, encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} conversations from {RAW_FILE}")

    # Build standardized examples (excluding adversarial)
    examples = build_examples(raw_data)
    logger.info(f"Built {len(examples)} non-adversarial QA examples")

    # Log category distribution
    cats = Counter(ex["question_type"] for ex in examples)
    logger.info(f"Category distribution: {dict(sorted(cats.items()))}")

    # Split
    train, val, test = create_splits(examples)
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Log split category distributions
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = Counter(ex["question_type"] for ex in split)
        logger.info(f"  {name}: {dict(sorted(dist.items()))}")

    # Save
    save_jsonl(train, OUTPUT_DIR / "locomo_train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "locomo_val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "locomo_test.jsonl")

    logger.info("LoCoMo preparation complete!")


if __name__ == "__main__":
    main()
