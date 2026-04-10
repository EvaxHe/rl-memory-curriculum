"""
End-to-end inference pipeline for Memory-R1.

Processes multi-session dialogues:
1. For each turn: Memory Manager decides CRUD operation
2. For each question: Answer Agent retrieves + reasons + answers

This pipeline is used for both evaluation and data preparation.
"""
import json
import logging
from typing import Optional
from dataclasses import dataclass

from src.memory.bank import MemoryBank
from src.agents.memory_manager import (
    build_mm_prompt, parse_mm_output, execute_mm_operation
)
from src.agents.answer_agent import build_aa_prompt, parse_aa_output

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the inference pipeline."""
    retrieval_top_k: int = 60       # Memory-R1 retrieves 60 candidates
    max_new_tokens_mm: int = 256    # Max tokens for Memory Manager
    max_new_tokens_aa: int = 512    # Max tokens for Answer Agent
    temperature_mm: float = 0.3     # Lower temp for structured output
    temperature_aa: float = 0.7     # Higher temp for reasoning


class MemoryR1Pipeline:
    """
    End-to-end Memory-R1 pipeline.

    In training mode, uses the base model.
    In eval mode, loads RL-trained checkpoints.
    """

    def __init__(self, model=None, tokenizer=None,
                 config: Optional[PipelineConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or PipelineConfig()
        self.memory_bank = MemoryBank()

    def reset_memory(self):
        """Clear memory bank for a new conversation."""
        self.memory_bank = MemoryBank()

    def process_turn(self, session_id: int, turn_id: int,
                     speaker: str, message: str) -> dict:
        """
        Process a single dialogue turn through the Memory Manager.
        Returns the operation performed.
        """
        # Build prompt
        messages = build_mm_prompt(
            self.memory_bank, session_id, turn_id, speaker, message
        )

        # Generate (placeholder — actual generation depends on model backend)
        raw_output = self._generate(messages, self.config.max_new_tokens_mm,
                                     self.config.temperature_mm)

        # Parse and execute
        operation = parse_mm_output(raw_output)
        result = execute_mm_operation(operation, self.memory_bank, session_id)
        self.memory_bank.advance_turn()

        return {
            "operation": operation,
            "result": result,
            "memory_size": self.memory_bank.size(),
        }

    def answer_question(self, question: str) -> dict:
        """
        Answer a question using the memory bank.
        1. Retrieve top-k memories
        2. Run Answer Agent
        3. Parse and return answer
        """
        # Retrieve
        retrieved = self.memory_bank.search(question,
                                             top_k=self.config.retrieval_top_k)

        # Build prompt
        messages = build_aa_prompt(question, retrieved)

        # Generate
        raw_output = self._generate(messages, self.config.max_new_tokens_aa,
                                     self.config.temperature_aa)

        # Parse
        parsed = parse_aa_output(raw_output)

        return {
            "answer": parsed["answer"],
            "reasoning": parsed["reasoning"],
            "selected_memories": parsed["selected_memories"],
            "num_retrieved": len(retrieved),
            "raw_output": raw_output,
        }

    def _generate(self, messages: list[dict], max_new_tokens: int,
                  temperature: float) -> str:
        """
        Generate text from the model.
        Override this method for different backends (vLLM, HF, API).
        """
        if self.model is None:
            raise RuntimeError(
                "No model loaded. Call load_model() or pass model to __init__."
            )

        # Default: HuggingFace transformers generate
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

        # Decode only the new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def process_conversation(pipeline: MemoryR1Pipeline,
                         conversation: dict) -> list[dict]:
    """
    Process an entire multi-session conversation and answer all questions.

    Expected conversation format (LoCoMo-style):
    {
        "conversation_id": "...",
        "sessions": [
            {"session_id": 1, "turns": [{"speaker": "...", "text": "..."}]},
            ...
        ],
        "questions": [
            {"question": "...", "answer": "...", "type": "single_hop"},
            ...
        ]
    }
    """
    pipeline.reset_memory()

    # Phase 1: Process all dialogue turns
    for session in conversation.get("sessions", []):
        sid = session["session_id"]
        for i, turn in enumerate(session["turns"]):
            pipeline.process_turn(
                session_id=sid,
                turn_id=i,
                speaker=turn["speaker"],
                message=turn["text"],
            )

    # Phase 2: Answer all questions
    results = []
    for q in conversation.get("questions", []):
        answer_result = pipeline.answer_question(q["question"])
        answer_result["gold_answer"] = q.get("answer", "")
        answer_result["question_type"] = q.get("type", "unknown")
        answer_result["question"] = q["question"]
        results.append(answer_result)

    return results
