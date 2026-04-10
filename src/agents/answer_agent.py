"""
Answer Agent for Memory-R1.

Given a question and retrieved memories, the Answer Agent:
1. Pre-selects the most relevant memories (distillation from 60 → ~5)
2. Reasons over selected memories to generate an answer

During training, fine-tuned with GRPO where reward = F1(answer, gold_answer).
"""
import re

from src.memory.entry import MemoryEntry
from src.common.prompts import AA_SYSTEM_PROMPT, AA_USER_TEMPLATE


def build_aa_prompt(question: str, retrieved_memories: list[MemoryEntry]) -> list[dict]:
    """Build the prompt for the Answer Agent."""
    if retrieved_memories:
        mem_lines = []
        for e in retrieved_memories:
            meta = f"(session {e.source_session}"
            if e.timestamp:
                meta += f", {e.timestamp}"
            meta += ")"
            mem_lines.append(f"- [{e.entry_id}] {e.content} {meta}")
        memories_str = "\n".join(mem_lines)
    else:
        memories_str = "No relevant memories found."

    user_msg = AA_USER_TEMPLATE.format(
        num_retrieved=len(retrieved_memories),
        memories=memories_str,
        question=question,
    )
    return [
        {"role": "system", "content": AA_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def parse_aa_output(raw_output: str) -> dict:
    """Parse Answer Agent output into structured components."""
    result = {
        "selected_memories": [],
        "reasoning": "",
        "answer": "",
    }

    # Extract selected memories
    sel_match = re.search(
        r"<selected_memories>\s*(.*?)\s*</selected_memories>",
        raw_output, re.DOTALL
    )
    if sel_match:
        ids_str = sel_match.group(1).strip()
        result["selected_memories"] = [
            s.strip() for s in ids_str.split(",") if s.strip()
        ]

    # Extract reasoning
    reason_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        raw_output, re.DOTALL
    )
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()

    # Extract answer
    ans_match = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        raw_output, re.DOTALL
    )
    if ans_match:
        result["answer"] = ans_match.group(1).strip()
    else:
        # Fallback: use the last non-empty line
        lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
        if lines:
            result["answer"] = lines[-1]

    return result


def extract_answer_from_completion(text: str) -> str:
    """Extract answer from AA output (XML tags or fallback to last line)."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""
