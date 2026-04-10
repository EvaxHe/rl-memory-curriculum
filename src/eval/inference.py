"""
Inference functions for evaluation — answer generation and MM processing.

Supports both HuggingFace and vLLM backends.
"""
import logging
import re

from src.common.prompts import AA_SYSTEM_PROMPT, AA_USER_TEMPLATE
from src.memory.bank import MemoryBank
from src.agents.memory_manager import build_mm_prompt, parse_mm_output, execute_mm_operation

logger = logging.getLogger(__name__)


def format_aa_prompt(question, retrieved_memories):
    """Format a single AA prompt (returns chat messages list)."""
    mem_str = "\n".join(f"- {m}" for m in retrieved_memories) \
        if retrieved_memories else "No relevant memories found."
    return [
        {"role": "system", "content": AA_SYSTEM_PROMPT},
        {"role": "user", "content": AA_USER_TEMPLATE.format(
            num_retrieved=len(retrieved_memories),
            memories=mem_str, question=question,
        )},
    ]


def extract_answer(raw_output):
    """Extract answer from model output."""
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw_output, re.DOTALL)
    if ans_match:
        return ans_match.group(1).strip()
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    return lines[-1] if lines else raw_output.strip()


def generate_answer(model, tokenizer, question, retrieved_memories,
                    max_new_tokens=512):
    """Generate an answer for a single question (fallback for non-batched)."""
    import torch
    messages = format_aa_prompt(question, retrieved_memories)
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=4096
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.3, do_sample=True, top_p=0.9,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return extract_answer(raw_output)


def generate_answers_batched(model, tokenizer, prompts_list,
                             max_new_tokens=1024, batch_size=8):
    """
    Batched answer generation. Significantly faster than one-at-a-time.
    prompts_list: list of chat message lists (one per question).
    Returns list of extracted answer strings.
    """
    import torch

    # Convert chat messages to text
    texts = []
    for messages in prompts_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    answers = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.3, do_sample=True, top_p=0.9,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].ne(tokenizer.pad_token_id).sum().item()
            new_tokens = output[input_len:]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers.append(extract_answer(raw_output))

    return answers


def generate_answers_vllm(llm, prompts_list, max_new_tokens=1024,
                          temperature=0.3, top_p=0.9):
    """
    Generate answers for all prompts using vLLM offline engine.
    vLLM handles continuous batching + PagedAttention internally.
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    texts = []
    for messages in prompts_list:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    logger.info(f"vLLM generating {len(texts)} prompts...")
    outputs = llm.generate(texts, sampling_params)

    answers = []
    for output in outputs:
        raw_output = output.outputs[0].text
        answers.append(extract_answer(raw_output))

    return answers


def run_mm_on_sessions(mm_model, mm_tokenizer, sessions, max_new_tokens=256):
    """
    Run trained Memory Manager on all dialogue turns to build a memory bank.
    Returns list of memory strings (same format as heuristic memories).
    """
    import torch

    bank = MemoryBank(use_embeddings=False)

    for session in sessions:
        sid = session.get("session_id", 0)
        for i, turn in enumerate(session.get("turns", [])):
            messages = build_mm_prompt(
                bank, session_id=sid, turn_id=i,
                speaker=turn["speaker"], message=turn["text"][:500],
            )
            text = mm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = mm_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=4096,
            ).to(mm_model.device)

            with torch.no_grad():
                outputs = mm_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=0.3, do_sample=True, top_p=0.9,
                )
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            raw_output = mm_tokenizer.decode(new_tokens, skip_special_tokens=True)

            operation = parse_mm_output(raw_output)
            execute_mm_operation(operation, bank, session_id=sid)
            bank.advance_turn()

    return [e.content for e in bank.get_all()]


def run_mm_on_sessions_vllm(mm_llm, sessions, max_new_tokens=256):
    """
    Run trained Memory Manager on all dialogue turns using vLLM.
    Sequential per conversation (each turn depends on memory state from prior turns).
    Returns list of memory strings.
    """
    from vllm import SamplingParams

    tokenizer = mm_llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.3, top_p=0.9, max_tokens=max_new_tokens,
    )
    bank = MemoryBank(use_embeddings=False)

    for session in sessions:
        sid = session.get("session_id", 0)
        for i, turn in enumerate(session.get("turns", [])):
            messages = build_mm_prompt(
                bank, session_id=sid, turn_id=i,
                speaker=turn["speaker"], message=turn["text"][:500],
            )
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            outputs = mm_llm.generate([text], sampling_params)
            raw_output = outputs[0].outputs[0].text

            operation = parse_mm_output(raw_output)
            execute_mm_operation(operation, bank, session_id=sid)
            bank.advance_turn()

    return [e.content for e in bank.get_all()]


def run_mm_all_conversations_vllm(mm_llm, conv_sessions, max_new_tokens=256):
    """
    Batch MM processing across all conversations using vLLM.

    Instead of processing each conversation sequentially (batch_size=1 per vLLM call),
    this processes all conversations step-wise: at each step, gather one pending turn
    from every active conversation and send them as a single batch to vLLM.

    conv_sessions: dict of {conv_id: sessions_list}
    Returns: dict of {conv_id: list[str]} (memory strings per conversation)
    """
    from vllm import SamplingParams

    tokenizer = mm_llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.3, top_p=0.9, max_tokens=max_new_tokens,
    )

    # Build per-conversation state: bank + flattened turn queue
    conv_banks = {}
    conv_turn_queues = {}
    for conv_id, sessions in conv_sessions.items():
        conv_banks[conv_id] = MemoryBank(use_embeddings=False)
        turns = []
        for session in sessions:
            sid = session.get("session_id", 0)
            for i, turn in enumerate(session.get("turns", [])):
                turns.append((sid, i, turn))
        conv_turn_queues[conv_id] = turns

    # Track position per conversation
    conv_positions = {conv_id: 0 for conv_id in conv_turn_queues}
    total_turns = sum(len(q) for q in conv_turn_queues.values())
    processed = 0
    step = 0

    while True:
        # Gather one pending turn from each active conversation
        batch_conv_ids = []
        batch_texts = []
        batch_sids = []

        for conv_id, pos in conv_positions.items():
            queue = conv_turn_queues[conv_id]
            if pos >= len(queue):
                continue
            sid, turn_idx, turn = queue[pos]
            bank = conv_banks[conv_id]
            messages = build_mm_prompt(
                bank, session_id=sid, turn_id=turn_idx,
                speaker=turn["speaker"], message=turn["text"][:500],
            )
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            batch_conv_ids.append(conv_id)
            batch_texts.append(text)
            batch_sids.append(sid)

        if not batch_texts:
            break

        step += 1
        if step % 10 == 1 or len(batch_texts) > 1:
            logger.info(
                f"  MM batch step {step}: {len(batch_texts)} prompts "
                f"({processed}/{total_turns} turns done)"
            )

        outputs = mm_llm.generate(batch_texts, sampling_params)

        for conv_id, sid, output in zip(batch_conv_ids, batch_sids, outputs):
            raw_output = output.outputs[0].text
            bank = conv_banks[conv_id]
            operation = parse_mm_output(raw_output)
            execute_mm_operation(operation, bank, session_id=sid)
            bank.advance_turn()
            conv_positions[conv_id] += 1
            processed += 1

    logger.info(f"  MM batched processing complete: {processed} turns in {step} steps")

    return {
        conv_id: [e.content for e in bank.get_all()]
        for conv_id, bank in conv_banks.items()
    }
