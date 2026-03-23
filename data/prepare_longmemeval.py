"""
Prepare LongMemEval benchmark data for Memory-R1 training.

LongMemEval (Wu et al., ICLR 2025): 500 questions across 6 categories:
  single-session-user (70), single-session-assistant (56),
  single-session-preference (30), multi-session (133),
  temporal-reasoning (133), knowledge-update (78)

We use longmemeval_s_cleaned.json (~115k tokens, ~40 sessions per question).

Raw data format:
  [{
    "question_id": "e47becba",
    "question_type": "single-session-user",
    "question": "What degree did I graduate with?",
    "question_date": "2023/05/30 (Tue) 23:40",
    "answer": "Business Administration",
    "answer_session_ids": ["answer_280352e9"],
    "haystack_dates": ["2023/05/20 (Sat) 02:21", ...],
    "haystack_session_ids": ["sharegpt_yywfIrx_0", ...],
    "haystack_sessions": [
      [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
      ...
    ]
  }]

Output format (JSONL):
  {"conversation_id", "sessions": [{"session_id", "date_time", "turns": [...]}],
   "question", "answer", "question_type", "question_date", "source_benchmark"}

We create a stratified train split: 10 questions per category (50 total
from the 5 non-abstention categories, plus any abstention variants).
"""
import json
import logging
import random
import subprocess
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

RAW_CANDIDATES = [
    Path("data/raw/longmemeval/data/longmemeval_s_cleaned.json"),
    Path("data/raw/longmemeval/data/longmemeval_s.json"),
    Path("data/raw/longmemeval/longmemeval_s_cleaned.json"),
    Path("data/raw/longmemeval/longmemeval_s.json"),
]
OUTPUT_DIR = Path("data/processed")

TRAIN_PER_CATEGORY = 10  # 10 per category


def convert_sessions(item: dict) -> list[dict]:
    """Convert LongMemEval's haystack_sessions to our standard format."""
    sessions = []
    haystack_sessions = item.get("haystack_sessions", [])
    haystack_dates = item.get("haystack_dates", [])
    haystack_ids = item.get("haystack_session_ids", [])

    for i, sess_turns in enumerate(haystack_sessions):
        turns = []
        for turn in sess_turns:
            turns.append({
                "speaker": turn.get("role", ""),
                "text": turn.get("content", ""),
            })
        sessions.append({
            "session_id": haystack_ids[i] if i < len(haystack_ids) else f"session_{i}",
            "date_time": haystack_dates[i] if i < len(haystack_dates) else "",
            "turns": turns,
        })

    return sessions


def build_examples(raw_data: list[dict]) -> list[dict]:
    """Convert raw LongMemEval format to our standard JSONL format."""
    examples = []
    for item in raw_data:
        sessions = convert_sessions(item)
        examples.append({
            "conversation_id": item["question_id"],
            "sessions": sessions,
            "question": item["question"],
            "answer": str(item["answer"]),
            "question_type": item["question_type"],
            "question_date": item.get("question_date", ""),
            "answer_session_ids": item.get("answer_session_ids", []),
            "source_benchmark": "longmemeval",
        })
    return examples


def create_stratified_split(examples: list[dict], seed: int = 42):
    """
    Stratified train/val/test split.
    Sample TRAIN_PER_CATEGORY from each category for training.
    Take a small val set, rest is test.
    """
    random.seed(seed)

    by_category = defaultdict(list)
    for ex in examples:
        by_category[ex["question_type"]].append(ex)

    logger.info("Category distribution:")
    for cat in sorted(by_category.keys()):
        logger.info(f"  {cat}: {len(by_category[cat])}")

    train = []
    remaining = []

    for cat, items in by_category.items():
        random.shuffle(items)
        n_train = min(TRAIN_PER_CATEGORY, len(items) // 2)
        train.extend(items[:n_train])
        remaining.extend(items[n_train:])

    random.shuffle(train)
    random.shuffle(remaining)

    # Small val split from remaining
    val_size = min(25, len(remaining) // 10)
    val = remaining[:val_size]
    test = remaining[val_size:]

    return train, val, test


def save_jsonl(data: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(data)} examples to {path}")


def resolve_raw_file() -> Path | None:
    """Find LongMemEval source file from common download/layout variants."""
    for candidate in RAW_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    raw_file = resolve_raw_file()
    if raw_file is None:
        clone_dir = Path("data/raw/longmemeval")
        if not clone_dir.exists():
            logger.info(f"Raw data not found. Cloning LongMemEval repo into {clone_dir}...")
            subprocess.run(
                ["git", "clone", "https://github.com/xiaowu0162/LongMemEval", str(clone_dir)],
                check=True,
            )
        # Download the cleaned dataset JSON if not present
        target_json = clone_dir / "data" / "longmemeval_s_cleaned.json"
        if not target_json.exists():
            target_json.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading longmemeval_s_cleaned.json into {target_json}...")
            subprocess.run(
                ["curl", "-L", "-o", str(target_json),
                 "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"],
                check=True,
            )
        raw_file = resolve_raw_file()
        if raw_file is None:
            logger.error("Raw file still not found after auto-download.")
            return

    with open(raw_file, encoding="utf-8") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} questions from {raw_file}")

    # Build standardized examples
    examples = build_examples(raw_data)
    logger.info(f"Built {len(examples)} examples")

    # Split
    train, val, test = create_stratified_split(examples)
    logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Log split distributions
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = Counter(ex["question_type"] for ex in split)
        logger.info(f"  {name}: {dict(sorted(dist.items()))}")

    # Save
    save_jsonl(train, OUTPUT_DIR / "longmemeval_train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "longmemeval_val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "longmemeval_test.jsonl")

    logger.info("LongMemEval preparation complete!")


if __name__ == "__main__":
    main()
