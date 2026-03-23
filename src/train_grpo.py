"""
GRPO training for Memory-R1 Answer Agent and Memory Manager using TRL.

Supports both LoRA and full fine-tuning, controlled by YAML config.
No code changes needed between configurations.

Usage:
    # Train Answer Agent with LoRA (default, ≥48GB GPU)
    python src/train_grpo.py --config configs/train_locomo_only.yaml --agent answer_agent

    # Train both agents with full FT (≥80GB GPU)
    python src/train_grpo.py --config configs/full_ft/train_locomo_only.yaml --agent both
"""
import argparse
import json
import logging
import re
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from datasets import Dataset
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================================
# Reward functions (passed to GRPOTrainer)
# ============================================================

def normalize_answer(text: str) -> str:
    """Normalize for fair comparison."""
    import string
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 reward signal."""
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
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Binary exact match reward signal (matches Memory-R1 paper)."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def extract_answer_from_completion(text: str) -> str:
    """Extract answer from AA output (XML tags or fallback to last line)."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


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


# ============================================================
# Data preparation for GRPOTrainer
# ============================================================

def load_training_data(data_path: str) -> list[dict]:
    """Load JSONL training data."""
    data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def build_heuristic_memory(example: dict) -> list[str]:
    """
    Build memory entries using simple heuristics (no RL).
    Used to create AA training prompts before MM is trained.
    """
    memories = []
    skip = {"hi", "hello", "hey", "thanks", "bye", "ok", "okay", "yes", "no"}

    for session in example.get("sessions", []):
        sid = session.get("session_id", 0)
        dt = session.get("date_time", "")
        for turn in session.get("turns", []):
            text = turn.get("text", "").strip()
            speaker = turn.get("speaker", "")
            words = text.lower().split()
            if len(words) > 5 and not any(w in skip for w in words[:2]):
                mem = f"{speaker}: {text[:300]}"
                if dt:
                    mem += f" (session {sid}, {dt})"
                memories.append(mem)

    return memories


def retrieve_memories_for_training(question: str, all_memories: list[str],
                                   top_k: int = 20) -> list[str]:
    """Retrieve memories for AA training prompts. Embedding with keyword fallback."""
    if not all_memories:
        return []

    # Try embedding retrieval first
    try:
        from src.retriever import embed_texts, search_numpy_fallback
        corpus_emb = embed_texts(all_memories)
        if corpus_emb is not None:
            query_emb = embed_texts([question])
            if query_emb is not None:
                _, indices = search_numpy_fallback(
                    query_emb[0], corpus_emb, top_k=min(top_k, len(all_memories))
                )
                return [all_memories[i] for i in indices if i < len(all_memories)]
    except Exception:
        pass

    # Keyword fallback
    q_words = set(question.lower().split())
    scored = []
    for mem in all_memories:
        overlap = len(q_words & set(mem.lower().split()))
        scored.append((overlap, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


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
        top_memories = retrieve_memories_for_training(question, all_memories, top_k=max_memories)

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


# ============================================================
# Prompt templates
# ============================================================

AA_SYSTEM_PROMPT = """You are an Answer Agent for a conversational AI assistant.
You have access to a memory bank containing facts from past conversations.

Given a question and retrieved memories, you must:
1. Select the most relevant memories for answering the question.
2. Reason step-by-step using the selected memories.
3. Provide a concise, accurate answer.

Output format:
<selected_memories>
[list the memory IDs or snippets you're using]
</selected_memories>
<reasoning>
[your step-by-step reasoning]
</reasoning>
<answer>
[your final answer - be concise]
</answer>"""

AA_USER_TEMPLATE = """## Retrieved Memories (top {num_retrieved})
{memories}

## Question
{question}

## Your Response:"""


MM_SYSTEM_PROMPT = """You are a Memory Manager for a conversational AI assistant.
Your job is to maintain an external memory bank by deciding what information to store,
update, or remove after each dialogue turn.

Given the current dialogue turn and existing memories, output a JSON operation:

Operations:
- ADD: Store new important information. Output: {{"op": "ADD", "content": "<fact to store>"}}
- UPDATE: Modify an existing memory. Output: {{"op": "UPDATE", "entry_id": "<id>", "content": "<updated fact>"}}
- DELETE: Remove outdated/incorrect memory. Output: {{"op": "DELETE", "entry_id": "<id>"}}
- NOOP: No memory change needed. Output: {{"op": "NOOP"}}

Rules:
1. Only ADD facts important for future conversations (preferences, events, relationships).
2. UPDATE when the user corrects or changes a previously stated fact.
3. DELETE when information is explicitly retracted.
4. NOOP for casual/greeting turns with no memorable information.
5. Output exactly ONE operation as valid JSON."""


# ============================================================
# Training functions
# ============================================================

def get_peft_config(config: dict):
    """Return LoRA config if use_lora is true, else None (full FT)."""
    use_lora = config["training"].get("use_lora", True)
    if not use_lora:
        logger.info("Full fine-tuning mode (no LoRA)")
        return None

    from peft import LoraConfig
    logger.info("LoRA mode (r=16, alpha=32)")
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )


def train_answer_agent(config: dict):
    """Train Answer Agent with GRPO via TRL."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_name = config["model"]["name"]
    exp_name = config["experiment"]["name"]
    data_path = config["data"]["train_file"]
    seed = config["experiment"]["seed"]
    reward_type = config.get("reward", {}).get("type", "f1")
    retrieval_top_k = config.get("retrieval", {}).get("top_k", 20)

    logger.info(f"=== Training Answer Agent: {exp_name} ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Reward: {reward_type}")
    logger.info(f"Retrieval top-k: {retrieval_top_k}")

    # Load data and prepare dataset
    train_data = load_training_data(data_path)
    dataset = prepare_aa_dataset(train_data, max_memories=retrieval_top_k)
    logger.info(f"Prepared {len(dataset)} training prompts")

    # LoRA or full FT
    peft_config = get_peft_config(config)

    # GRPO training config
    aa_epochs = config["training"].get("aa_epochs", 2)
    group_size = config["training"].get("group_size", 4)
    batch_size = config["training"].get("batch_size", 1)
    grad_accum = config["training"].get("gradient_accumulation_steps", 4)
    lr = config["training"].get("learning_rate", 5e-6)
    max_completion = config["training"].get("aa_max_completion_length", 512)
    use_grad_ckpt = config["training"].get("gradient_checkpointing", False)

    output_dir = f"checkpoints/{exp_name}/answer_agent"

    # Validate: generation_batch_size = batch_size * grad_accum must be divisible by num_generations
    gen_batch = batch_size * grad_accum
    if gen_batch % group_size != 0:
        logger.warning(f"generation_batch_size ({gen_batch}) not divisible by group_size ({group_size}). "
                       f"Adjusting group_size to {gen_batch}.")
        group_size = gen_batch

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f"{exp_name}_aa",
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_ckpt,
        num_generations=group_size,
        max_completion_length=max_completion,
        num_train_epochs=aa_epochs,
        save_steps=50,
        save_total_limit=2,
        max_grad_norm=0.1,
        seed=seed,
        report_to="none",
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build reward functions
    aa_reward = make_aa_reward_func(reward_type)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[aa_reward, format_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training metadata
    use_lora = config["training"].get("use_lora", True)
    meta = {
        "model": model_name,
        "experiment": exp_name,
        "agent": "answer_agent",
        "epochs": aa_epochs,
        "group_size": group_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": use_grad_ckpt,
        "num_examples": len(dataset),
        "learning_rate": lr,
        "reward_type": reward_type,
        "retrieval_top_k": retrieval_top_k,
        "max_completion_length": max_completion,
        "use_lora": use_lora,
        "lora_r": 16 if use_lora else None,
        "lora_alpha": 32 if use_lora else None,
    }
    with open(f"{output_dir}/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"AA training complete. Saved to {output_dir}")


def train_memory_manager(config: dict):
    """
    Train Memory Manager with GRPO via TRL.

    Single-turn: each prompt is a dialogue turn + current memory state.
    Reward = format correctness of CRUD operation.
    Full multi-turn training with downstream QA reward is future work.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    model_name = config["model"]["name"]
    exp_name = config["experiment"]["name"]
    data_path = config["data"]["train_file"]
    seed = config["experiment"]["seed"]

    logger.info(f"=== Training Memory Manager: {exp_name} ===")

    # Load training data
    train_data = load_training_data(data_path)

    # Prepare MM dataset: each example is a turn + memory state
    mm_prompts = []
    mm_answers = []

    for ex in train_data:
        memories = build_heuristic_memory(ex)
        mem_str = "\n".join(f"- [{i}] {m}" for i, m in enumerate(memories[-10:])) \
            if memories else "No memories stored."

        for session in ex.get("sessions", [])[:2]:
            sid = session.get("session_id", 0)
            for i, turn in enumerate(session.get("turns", [])[:5]):
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

    dataset = Dataset.from_dict({"prompt": mm_prompts, "answer": mm_answers})
    logger.info(f"Prepared {len(dataset)} MM training prompts")

    # MM reward: valid JSON format + correct CRUD structure
    def mm_format_reward(completions, **kwargs) -> list[float]:
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

    peft_config = get_peft_config(config)

    mm_epochs = config["training"].get("mm_epochs", 2)
    group_size = config["training"].get("group_size", 4)
    batch_size = config["training"].get("batch_size", 1)
    grad_accum = config["training"].get("gradient_accumulation_steps", 4)
    lr = config["training"].get("learning_rate", 5e-6)
    max_completion = config["training"].get("mm_max_completion_length", 256)
    use_grad_ckpt = config["training"].get("gradient_checkpointing", False)
    output_dir = f"checkpoints/{exp_name}/memory_manager"

    gen_batch = batch_size * grad_accum
    if gen_batch % group_size != 0:
        logger.warning(f"generation_batch_size ({gen_batch}) not divisible by group_size ({group_size}). "
                       f"Adjusting group_size to {gen_batch}.")
        group_size = gen_batch

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=f"{exp_name}_mm",
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        bf16=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_ckpt,
        num_generations=group_size,
        max_completion_length=max_completion,
        num_train_epochs=mm_epochs,
        save_steps=300,
        save_total_limit=2,
        max_grad_norm=0.1,
        seed=seed,
        report_to="none",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[mm_format_reward],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    logger.info("Starting MM GRPO training...")
    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    use_lora = config["training"].get("use_lora", True)
    meta = {
        "model": model_name,
        "experiment": exp_name,
        "agent": "memory_manager",
        "epochs": mm_epochs,
        "group_size": group_size,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "gradient_checkpointing": use_grad_ckpt,
        "num_examples": len(dataset),
        "learning_rate": lr,
        "max_completion_length": max_completion,
        "use_lora": use_lora,
        "lora_r": 16 if use_lora else None,
        "lora_alpha": 32 if use_lora else None,
    }
    with open(f"{output_dir}/training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"MM training complete. Saved to {output_dir}")


# ============================================================
# Main
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Memory-R1 with GRPO")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--agent", type=str, default="both",
                        choices=["memory_manager", "answer_agent", "both"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    config = load_config(args.config)

    exp_name = config["experiment"]["name"]
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"LoRA: {config['training'].get('use_lora', True)}")
    logger.info(f"Reward: {config.get('reward', {}).get('type', 'f1')}")

    if args.agent in ("answer_agent", "both"):
        logger.info("=== Training Answer Agent ===")
        train_answer_agent(config)

    if args.agent in ("memory_manager", "both"):
        logger.info("=== Training Memory Manager ===")
        train_memory_manager(config)

    logger.info(f"Training complete for {exp_name}")


if __name__ == "__main__":
    main()
