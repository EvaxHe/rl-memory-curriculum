"""
Dataset preparation for GRPO training.
"""
import json
import logging

import numpy as np
from datasets import Dataset

from src.common.prompts import AA_SYSTEM_PROMPT, AA_USER_TEMPLATE, MM_SYSTEM_PROMPT
from src.memory.heuristic import build_heuristic_memory, retrieve_memories, SKIP_WORDS

logger = logging.getLogger(__name__)


def load_training_data(data_path: str) -> list[dict]:
    """Load JSONL training data."""
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def prepare_aa_dataset(train_data: list[dict], max_memories: int = 20) -> Dataset:
    """
    Prepare dataset for Answer Agent GRPO training.
    max_memories controlled by config retrieval.top_k (default 60).
    """
    prompts = []
    answers = []

    for ex in train_data:
        all_memories = build_heuristic_memory(ex)
        question = ex["question"]
        top_memories = retrieve_memories(question, all_memories, top_k=max_memories)

        if not top_memories:
            top_memories = all_memories[-max_memories:]

        mem_str = "\n".join(f"- {m}" for m in top_memories) if top_memories else "No relevant memories found."

        prompt = [
            {"role": "system", "content": AA_SYSTEM_PROMPT},
            {"role": "user", "content": AA_USER_TEMPLATE.format(
                num_retrieved=len(top_memories),
                memories=mem_str,
                question=question,
            )},
        ]

        prompts.append(prompt)
        answers.append(str(ex["answer"]))

    return Dataset.from_dict({"prompt": prompts, "answer": answers})


def prepare_mm_dataset(train_data: list[dict], config: dict) -> Dataset:
    """
    Prepare dataset for Memory Manager GRPO training.

    Sliding window across full conversation with evolving memory state.
    Samples turns from early/mid/late conversation, accumulates memories per turn.
    """
    max_mm_prompts_per_example = config["training"].get("max_mm_prompts_per_example", 25)

    mm_prompts = []
    mm_answers = []

    for ex in train_data:
        evolving_memories = []
        prompts_for_ex = 0

        for session in ex.get("sessions", []):
            sid = session.get("session_id", 0)
            dt = session.get("date_time", "")
            turns = session.get("turns", [])

            # Sample turns spread across the session (early, mid, late)
            if len(turns) > 5:
                indices = np.linspace(0, len(turns) - 1, 5, dtype=int).tolist()
            else:
                indices = list(range(len(turns)))

            for i in indices:
                if prompts_for_ex >= max_mm_prompts_per_example:
                    break
                turn = turns[i]

                # Build prompt with CURRENT evolving memory state (last 20)
                if evolving_memories:
                    mem_str = "\n".join(
                        f"- [{j}] {m}" for j, m in enumerate(evolving_memories[-20:])
                    )
                else:
                    mem_str = "No memories stored."

                prompt = [
                    {"role": "system", "content": MM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"""## Current Memories
{mem_str}

## Current Dialogue Turn
Session {sid}, Turn {i}:
Speaker: {turn['speaker']}
Message: {turn['text'][:500]}

## Your Decision (output valid JSON):"""},
                ]
                mm_prompts.append(prompt)
                mm_answers.append(str(ex["answer"]))
                prompts_for_ex += 1

                # Simulate heuristic ADD so next turn sees updated state
                text = turn.get("text", "").strip()
                speaker = turn.get("speaker", "")
                words = text.lower().split()
                if len(words) > 5 and not any(w in SKIP_WORDS for w in words[:2]):
                    mem = f"{speaker}: {text[:300]}"
                    if dt:
                        mem += f" (session {sid}, {dt})"
                    evolving_memories.append(mem)

            if prompts_for_ex >= max_mm_prompts_per_example:
                break

    dataset = Dataset.from_dict({"prompt": mm_prompts, "answer": mm_answers})
    logger.info(f"Prepared {len(dataset)} MM training prompts")
    return dataset
