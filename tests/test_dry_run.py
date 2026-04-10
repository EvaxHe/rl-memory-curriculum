"""
Local dry-run: exercises the full pipeline without a GPU.

Tests:
1. Load processed data → verify format
2. Process turns through Memory Manager (heuristic, no LLM)
3. Build memory bank → verify CRUD ops work
4. Retrieve memories for questions → verify retrieval
5. Generate dummy answers → compute rewards/metrics
6. Run eval metrics → verify aggregation + table generation

This catches integration bugs before we burn GPU hours.

Usage:
    python -m tests.test_dry_run
"""
import json
import logging
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.bank import MemoryBank
from src.memory.entry import MemoryEntry
from src.agents.memory_manager import build_mm_prompt, parse_mm_output, execute_mm_operation
from src.agents.answer_agent import build_aa_prompt, parse_aa_output
from src.common.scoring import token_f1, bleu1, exact_match, compute_reward
from src.eval.metrics import evaluate_predictions, format_results_table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ============================================================
# Test 1: Data loading and format verification
# ============================================================

def test_data_loading():
    logger.info("=" * 60)
    logger.info("TEST 1: Data loading and format verification")
    logger.info("=" * 60)

    expected_files = {
        "locomo_train.jsonl": 152,
        "locomo_val.jsonl": 81,
        "locomo_test.jsonl": 1307,
        "longmemeval_train.jsonl": 60,
        "longmemeval_val.jsonl": 25,
        "longmemeval_test.jsonl": 415,
        "mixed_train.jsonl": 212,
        "mixed_val.jsonl": 106,
    }

    for fname, expected_count in expected_files.items():
        path = DATA_DIR / fname
        assert path.exists(), f"Missing: {path}"
        data = load_jsonl(path)
        assert len(data) == expected_count, \
            f"{fname}: expected {expected_count}, got {len(data)}"

        # Verify required fields
        ex = data[0]
        for field in ["conversation_id", "sessions", "question", "answer",
                       "question_type", "source_benchmark"]:
            assert field in ex, f"{fname}: missing field '{field}'"

        # Verify sessions structure
        assert len(ex["sessions"]) > 0, f"{fname}: empty sessions"
        sess = ex["sessions"][0]
        assert "session_id" in sess, f"{fname}: session missing session_id"
        assert "turns" in sess, f"{fname}: session missing turns"
        assert len(sess["turns"]) > 0, f"{fname}: empty turns"
        turn = sess["turns"][0]
        assert "speaker" in turn, f"{fname}: turn missing speaker"
        assert "text" in turn, f"{fname}: turn missing text"

        logger.info(f"  ✓ {fname}: {len(data)} examples, format OK")

    logger.info("TEST 1 PASSED\n")
    return True


# ============================================================
# Test 2: Memory Bank CRUD operations
# ============================================================

def test_memory_bank():
    logger.info("=" * 60)
    logger.info("TEST 2: Memory Bank CRUD operations")
    logger.info("=" * 60)

    bank = MemoryBank(use_embeddings=False)

    # ADD
    id1 = bank.add("User likes hiking", source_session=1, timestamp="May 2023")
    id2 = bank.add("User has a dog named Buddy", source_session=1)
    id3 = bank.add("User is vegetarian", source_session=2)
    assert bank.size() == 3, f"Expected 3, got {bank.size()}"
    logger.info(f"  ✓ ADD: 3 entries created ({id1}, {id2}, {id3})")

    # UPDATE
    success = bank.update(id3, "User started eating fish recently")
    assert success, "UPDATE should succeed"
    entry = bank.get_by_id(id3)
    assert "fish" in entry.content, "UPDATE content not applied"
    logger.info(f"  ✓ UPDATE: {id3} updated")

    # DELETE
    success = bank.delete(id1)
    assert success, "DELETE should succeed"
    assert bank.size() == 2, f"Expected 2 after delete, got {bank.size()}"
    logger.info(f"  ✓ DELETE: {id1} removed, size={bank.size()}")

    # NOOP
    bank.noop()
    assert bank.size() == 2, "NOOP should not change size"
    logger.info("  ✓ NOOP: no change")

    # Search (keyword fallback)
    results = bank.search_keyword("dog", top_k=5)
    assert len(results) > 0, "Search should find 'dog'"
    assert "dog" in results[0].content.lower() or "buddy" in results[0].content.lower()
    logger.info(f"  ✓ SEARCH: found {len(results)} results for 'dog'")

    # Serialization
    json_str = bank.to_json()
    bank2 = MemoryBank.from_json(json_str)
    assert bank2.size() == bank.size(), "Deserialized bank size mismatch"
    logger.info("  ✓ SERIALIZATION: round-trip OK")

    # format_for_prompt
    prompt_str = bank.format_for_prompt()
    assert len(prompt_str) > 0, "format_for_prompt should return non-empty"
    logger.info(f"  ✓ FORMAT: prompt has {len(prompt_str)} chars")

    logger.info("TEST 2 PASSED\n")
    return True


# ============================================================
# Test 3: Memory Manager prompt building and parsing
# ============================================================

def test_memory_manager():
    logger.info("=" * 60)
    logger.info("TEST 3: Memory Manager prompt + parse + execute")
    logger.info("=" * 60)

    bank = MemoryBank(use_embeddings=False)

    # Build prompt
    messages = build_mm_prompt(bank, session_id=1, turn_id=0,
                                speaker="Caroline",
                                message="I just adopted a dog named Buddy!")
    assert len(messages) == 2, "Should have system + user messages"
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Buddy" in messages[1]["content"]
    logger.info("  ✓ PROMPT: built correctly")

    # Parse various outputs
    test_cases = [
        ('{"op": "ADD", "content": "User adopted a dog named Buddy"}',
         {"op": "ADD", "content": "User adopted a dog named Buddy"}),
        ('```json\n{"op": "NOOP"}\n```', {"op": "NOOP"}),
        ('I think we should add this. {"op": "ADD", "content": "test"}',
         {"op": "ADD", "content": "test"}),
        ('garbage output with no json', {"op": "NOOP"}),
        ('{"op": "UPDATE", "entry_id": "abc123", "content": "updated"}',
         {"op": "UPDATE", "entry_id": "abc123", "content": "updated"}),
        ('{"op": "DELETE", "entry_id": "abc123"}',
         {"op": "DELETE", "entry_id": "abc123"}),
    ]

    for raw, expected in test_cases:
        parsed = parse_mm_output(raw)
        assert parsed["op"] == expected["op"], \
            f"Parse '{raw[:40]}...': expected op={expected['op']}, got {parsed['op']}"
    logger.info(f"  ✓ PARSE: {len(test_cases)} test cases passed")

    # Execute operations
    result = execute_mm_operation(
        {"op": "ADD", "content": "User adopted a dog named Buddy"},
        bank, session_id=1
    )
    assert "ADD" in result
    assert bank.size() == 1
    logger.info(f"  ✓ EXECUTE ADD: {result}")

    bank.advance_turn()

    result = execute_mm_operation({"op": "NOOP"}, bank, session_id=1)
    assert "NOOP" in result
    assert bank.size() == 1
    logger.info(f"  ✓ EXECUTE NOOP: {result}")

    logger.info("TEST 3 PASSED\n")
    return True


# ============================================================
# Test 4: Answer Agent prompt building and parsing
# ============================================================

def test_answer_agent():
    logger.info("=" * 60)
    logger.info("TEST 4: Answer Agent prompt + parse")
    logger.info("=" * 60)

    # Create some memories
    entries = [
        MemoryEntry("a1", "User adopted a dog named Buddy", 1, "May 2023", 0, 0),
        MemoryEntry("b2", "User likes hiking on weekends", 1, "May 2023", 1, 1),
        MemoryEntry("c3", "User is vegetarian", 2, "June 2023", 5, 5),
    ]

    # Build prompt
    messages = build_aa_prompt("What is the user's dog's name?", entries)
    assert len(messages) == 2
    assert "Buddy" in messages[1]["content"]
    assert "dog" in messages[1]["content"].lower()
    logger.info("  ✓ PROMPT: built correctly with 3 memories")

    # Parse well-formed output
    good_output = """<selected_memories>
a1, b2
</selected_memories>
<reasoning>
The user mentioned adopting a dog named Buddy in session 1.
</reasoning>
<answer>
Buddy
</answer>"""
    parsed = parse_aa_output(good_output)
    assert parsed["answer"] == "Buddy"
    assert "a1" in parsed["selected_memories"]
    assert len(parsed["reasoning"]) > 0
    logger.info(f"  ✓ PARSE (well-formed): answer='{parsed['answer']}'")

    # Parse messy output (no XML tags)
    messy_output = "The dog's name is Buddy."
    parsed = parse_aa_output(messy_output)
    assert "Buddy" in parsed["answer"]
    logger.info(f"  ✓ PARSE (fallback): answer='{parsed['answer']}'")

    # Parse empty output
    parsed = parse_aa_output("")
    assert parsed["answer"] == ""
    logger.info("  ✓ PARSE (empty): handled gracefully")

    logger.info("TEST 4 PASSED\n")
    return True


# ============================================================
# Test 5: Reward functions
# ============================================================

def test_rewards():
    logger.info("=" * 60)
    logger.info("TEST 5: Reward functions")
    logger.info("=" * 60)

    # F1
    assert token_f1("Buddy", "Buddy") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("Buddy the dog", "Buddy") > 0
    assert token_f1("completely wrong", "Buddy") == 0.0
    f1_partial = token_f1("The dog is named Buddy", "Buddy the golden retriever")
    assert 0 < f1_partial < 1
    logger.info(f"  ✓ F1: exact=1.0, partial={f1_partial:.3f}, wrong=0.0")

    # BLEU-1
    assert bleu1("Buddy", "Buddy") == 1.0
    assert bleu1("completely wrong answer", "Buddy") == 0.0
    logger.info("  ✓ BLEU-1: exact=1.0, wrong=0.0")

    # Exact Match
    assert exact_match("Buddy", "Buddy") == 1.0
    assert exact_match("buddy", "Buddy") == 1.0  # case insensitive
    assert exact_match("The Buddy", "Buddy") == 1.0  # "the" is stripped as article
    assert exact_match("completely wrong", "Buddy") == 0.0
    logger.info("  ✓ EM: exact=1.0, case-insensitive=1.0, article-stripped=1.0")

    # Combined
    combined = compute_reward("Buddy", "Buddy", "combined")
    assert abs(combined - 1.0) < 1e-6, f"Expected ~1.0, got {combined}"
    logger.info(f"  ✓ COMBINED: perfect={combined:.6f}")

    logger.info("TEST 5 PASSED\n")
    return True


# ============================================================
# Test 6: Heuristic pipeline on real data
# ============================================================

def heuristic_memory_manager(bank: MemoryBank, session_id: int,
                              speaker: str, text: str):
    """Simple heuristic MM: ADD non-trivial user turns."""
    skip = {"hi", "hello", "hey", "thanks", "bye", "ok", "okay",
            "yes", "no", "yeah", "sure", "right"}
    words = text.lower().split()
    if len(words) > 5 and not any(w in skip for w in words[:2]):
        bank.add(f"{speaker}: {text[:200]}", source_session=session_id)
    bank.advance_turn()


def test_heuristic_pipeline():
    logger.info("=" * 60)
    logger.info("TEST 6: Heuristic pipeline on real data (1 example each)")
    logger.info("=" * 60)

    for benchmark in ["locomo", "longmemeval"]:
        path = DATA_DIR / f"{benchmark}_train.jsonl"
        data = load_jsonl(path)
        ex = data[0]

        logger.info(f"\n  --- {benchmark} ---")
        logger.info(f"  conversation_id: {ex['conversation_id']}")
        logger.info(f"  num sessions: {len(ex['sessions'])}")
        total_turns = sum(len(s["turns"]) for s in ex["sessions"])
        logger.info(f"  total turns: {total_turns}")

        # Process turns with heuristic MM
        bank = MemoryBank(use_embeddings=False)
        for session in ex["sessions"]:
            sid = session["session_id"]
            for turn in session["turns"]:
                heuristic_memory_manager(
                    bank, sid, turn["speaker"], turn["text"]
                )

        logger.info(f"  memories after processing: {bank.size()}")

        # Retrieve for the question
        question = ex["question"]
        gold = ex["answer"]
        retrieved = bank.search_keyword(question, top_k=10)
        logger.info(f"  question: {question}")
        logger.info(f"  gold answer: {gold}")
        logger.info(f"  retrieved {len(retrieved)} memories")

        if retrieved:
            logger.info(f"  top memory: {retrieved[0].content[:100]}...")

        # Dummy answer (use first retrieved memory content as proxy)
        dummy_answer = retrieved[0].content if retrieved else ""
        f1 = token_f1(dummy_answer, gold)
        logger.info(f"  dummy F1: {f1:.3f}")

    logger.info("\nTEST 6 PASSED\n")
    return True


# ============================================================
# Test 7: Eval metrics aggregation
# ============================================================

def test_eval_metrics():
    logger.info("=" * 60)
    logger.info("TEST 7: Eval metrics aggregation + table generation")
    logger.info("=" * 60)

    # Create dummy predictions
    predictions = [
        {"answer": "Buddy", "gold_answer": "Buddy",
         "question_type": "single_hop", "source_benchmark": "locomo"},
        {"answer": "hiking", "gold_answer": "hiking and swimming",
         "question_type": "single_hop", "source_benchmark": "locomo"},
        {"answer": "May 2023", "gold_answer": "7 May 2023",
         "question_type": "temporal", "source_benchmark": "locomo"},
        {"answer": "wrong answer", "gold_answer": "correct answer",
         "question_type": "multi_hop", "source_benchmark": "locomo"},
        {"answer": "Business Administration", "gold_answer": "Business Administration",
         "question_type": "single-session-user", "source_benchmark": "longmemeval"},
        {"answer": "8 meals", "gold_answer": "8 meals",
         "question_type": "multi-session", "source_benchmark": "longmemeval"},
    ]

    results = evaluate_predictions(predictions)

    assert "overall" in results
    assert "per_type" in results
    assert "per_benchmark" in results
    assert results["num_examples"] == 6

    overall = results["overall"]
    assert 0 < overall["f1"] <= 1.0
    assert "n" in overall and overall["n"] == 6
    logger.info(f"  ✓ Overall: F1={overall['f1']:.3f}, "
                f"BLEU={overall['bleu1']:.3f}, EM={overall['exact_match']:.3f}")

    # Per type
    assert "single_hop" in results["per_type"]
    assert "temporal" in results["per_type"]
    logger.info(f"  ✓ Per-type: {len(results['per_type'])} types")
    for qtype, m in sorted(results["per_type"].items()):
        logger.info(f"    {qtype}: F1={m['f1']:.3f} (n={m['n']})")

    # Per benchmark
    assert "locomo" in results["per_benchmark"]
    assert "longmemeval" in results["per_benchmark"]
    logger.info(f"  ✓ Per-benchmark: {len(results['per_benchmark'])} benchmarks")

    # Table generation
    table = format_results_table(results, "dry-run-test")
    assert "F1=" in table
    assert "single_hop" in table
    logger.info(f"  ✓ Table generated ({len(table)} chars)")
    logger.info(f"\n{table}")

    logger.info("\nTEST 7 PASSED\n")
    return True


# ============================================================
# Test 8: Full pipeline simulation (no model)
# ============================================================

def test_full_pipeline_simulation():
    logger.info("=" * 60)
    logger.info("TEST 8: Full pipeline simulation across train splits")
    logger.info("=" * 60)

    for split_name in ["locomo_train", "longmemeval_train", "mixed_train"]:
        path = DATA_DIR / f"{split_name}.jsonl"
        data = load_jsonl(path)

        # Process first 3 examples
        predictions = []
        for ex in data[:3]:
            bank = MemoryBank(use_embeddings=False)

            # Heuristic MM
            for session in ex["sessions"]:
                sid = session["session_id"]
                for turn in session["turns"]:
                    heuristic_memory_manager(
                        bank, sid, turn["speaker"], turn["text"]
                    )

            # Retrieve + dummy answer
            retrieved = bank.search_keyword(ex["question"], top_k=10)
            dummy_answer = retrieved[0].content.split(": ", 1)[-1][:100] if retrieved else ""

            predictions.append({
                "answer": dummy_answer,
                "gold_answer": ex["answer"],
                "question_type": ex["question_type"],
                "source_benchmark": ex["source_benchmark"],
                "question": ex["question"],
            })

        results = evaluate_predictions(predictions)
        overall = results["overall"]
        logger.info(f"  {split_name} (3 examples): "
                    f"F1={overall['f1']:.3f}, BLEU={overall['bleu1']:.3f}, "
                    f"EM={overall['exact_match']:.3f}")

    logger.info("\nTEST 8 PASSED\n")
    return True


# ============================================================
# Main
# ============================================================

def main():
    tests = [
        ("Data Loading", test_data_loading),
        ("Memory Bank", test_memory_bank),
        ("Memory Manager", test_memory_manager),
        ("Answer Agent", test_answer_agent),
        ("Rewards", test_rewards),
        ("Heuristic Pipeline", test_heuristic_pipeline),
        ("Eval Metrics", test_eval_metrics),
        ("Full Pipeline Sim", test_full_pipeline_simulation),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"FAILED: {name} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("=" * 60)
    logger.info(f"DRY RUN COMPLETE: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
