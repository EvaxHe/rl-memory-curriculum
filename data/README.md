# Data Preparation

## Benchmarks

| Benchmark | Source | Train Split | Test Split | Categories |
|-----------|--------|-------------|------------|------------|
| LoCoMo | [snap-research/locomo](https://github.com/snap-research/locomo) | 152 QA pairs | 1307 QA pairs | single-hop, multi-hop, temporal, open-domain |
| LongMemEval | [xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval) | 60 QA pairs (our split) | 415 QA pairs | single-session-user, single-session-assistant, single-session-preference, multi-session, temporal-reasoning, knowledge-update |
| MSC | [ParlAI](https://parl.ai/projects/msc/) | — | ~500 QA pairs | multi-session chat |

## Data Format

All processed data is in JSONL format. Each line:

```json
{
    "conversation_id": "locomo_1",
    "sessions": [
        {
            "session_id": 1,
            "turns": [
                {"speaker": "User", "text": "I just adopted a dog named Buddy!"},
                {"speaker": "Assistant", "text": "That's wonderful! What breed?"}
            ]
        }
    ],
    "question": "What is the name of the user's dog?",
    "answer": "Buddy",
    "question_type": "single_hop",
    "source_benchmark": "locomo"
}
```

## Preparation Steps

Processed data is not committed to GitHub. First clone the raw data repos, then
run the prep scripts:

```bash
# 0. Clone raw data (only needed if data/processed/ is empty)
git clone https://github.com/snap-research/locomo data/raw/locomo
git clone https://github.com/xiaowu0162/LongMemEval data/raw/longmemeval

# 0.1 Download LongMemEval data file (the repo clone alone does not include it)
curl -L -o data/raw/longmemeval/data/longmemeval_s_cleaned.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json

# 1. Process LoCoMo (152 train / 81 val / 1307 test)
python data/prepare_locomo.py

# 2. Process LongMemEval (60 train / 25 val / 415 test)
python data/prepare_longmemeval.py

# 3. Create mixed training set (212 train / 106 val)
python data/prepare_mixed.py
```

## LongMemEval Train/Test Split Methodology

We create a stratified train split from LongMemEval by sampling 10 questions
per category (60 total from 6 categories), preserving category balance.
25 questions are held out for validation; the remaining 415 form the test set.

Justification: Memory-R1 itself derives training data from LoCoMo's evaluation
set using a similar approach (152 train / 81 val / 1307 test). RL training
learns a memory management *policy*, not answer memorization, so data leakage
risk is minimal. See Section 3.2 of our paper for full discussion.
