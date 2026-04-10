"""
LLM-as-Judge evaluation using OpenAI-compatible API.

Supports any OpenAI-compatible endpoint (OpenAI, Anthropic via proxy,
local vLLM, etc.) via environment variables.

Set OPENAI_API_KEY and optionally OPENAI_BASE_URL before running.
"""
import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's answer
about a user's past conversations.

Question: {question}
Gold Answer: {gold_answer}
Predicted Answer: {predicted_answer}

Rate the predicted answer on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially relevant but mostly incorrect
3 = Captures some correct information but misses key details
4 = Mostly correct with minor omissions
5 = Fully correct and complete

Output ONLY a JSON object: {{"score": <1-5>, "reason": "<brief explanation>"}}"""


def create_client(model: str = "gpt-4o-mini"):
    """Create OpenAI-compatible client."""
    try:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. LLM-as-Judge disabled.")
            return None
        base_url = os.environ.get("OPENAI_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client
    except Exception as e:
        logger.warning(f"Could not create OpenAI client: {e}")
        return None


def judge_single(client, question: str, gold_answer: str,
                 predicted_answer: str,
                 model: str = "gpt-4o-mini") -> dict:
    """
    Judge a single prediction using an OpenAI-compatible API.
    Returns {"score": int, "reason": str}.
    """
    if client is None:
        return {"score": 0, "reason": "No API client available"}

    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        predicted_answer=predicted_answer,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0,
        )
        text = response.choices[0].message.content

        match = re.search(r"\{[^}]+\}", text)
        if match:
            return json.loads(match.group())
        return {"score": 0, "reason": f"Could not parse: {text[:100]}"}

    except Exception as e:
        logger.error(f"Judge API call failed: {e}")
        return {"score": 0, "reason": str(e)}


def judge_batch(predictions: list[dict],
                model: str = "gpt-4o-mini") -> list[dict]:
    """
    Judge a batch of predictions. Adds 'judge_score' and 'judge_reason'
    to each prediction dict.
    """
    client = create_client(model)
    if client is None:
        logger.warning("Skipping LLM-as-Judge (no API client)")
        return predictions

    for i, pred in enumerate(predictions):
        result = judge_single(
            client,
            question=pred.get("question", ""),
            gold_answer=pred.get("gold_answer", ""),
            predicted_answer=pred.get("answer", ""),
            model=model,
        )
        pred["judge_score"] = result.get("score", 0)
        pred["judge_reason"] = result.get("reason", "")

        if (i + 1) % 50 == 0:
            logger.info(f"Judged {i + 1}/{len(predictions)} predictions")

    scores = [p["judge_score"] for p in predictions if p["judge_score"] > 0]
    if scores:
        mean_score = sum(scores) / len(scores)
        logger.info(f"Mean judge score: {mean_score:.2f} (n={len(scores)})")

    return predictions
