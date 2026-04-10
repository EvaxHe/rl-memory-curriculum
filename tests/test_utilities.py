"""
Unit tests for pure utility functions — no ML models, no data files, no GPU.

Covers:
  - src/common/scoring.py     (normalize, F1, BLEU-1, EM, compute_reward)
  - src/common/config.py      (YAML loading)
  - src/common/prompts.py     (template placeholders)
  - src/memory/entry.py       (MemoryEntry dataclass)
  - src/memory/bank.py        (MemoryBank CRUD, keyword search, serialization)
  - src/memory/heuristic.py   (heuristic memory building, SKIP_WORDS)
  - src/agents/answer_agent.py (prompt building, XML parsing)
  - src/agents/memory_manager.py (prompt building, JSON parsing, execute)

Usage:
    uv run python -m pytest tests/test_utilities.py -v
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.common.scoring import (
    normalize_answer,
    token_f1,
    bleu1,
    exact_match,
    compute_reward,
)
from src.common.config import load_config
from src.common.prompts import (
    AA_SYSTEM_PROMPT,
    AA_USER_TEMPLATE,
    MM_SYSTEM_PROMPT,
    MM_USER_TEMPLATE,
)
from src.memory.entry import MemoryEntry
from src.memory.bank import MemoryBank
from src.memory.heuristic import (
    build_heuristic_memory,
    build_heuristic_memories,
    SKIP_WORDS,
)
from src.agents.answer_agent import (
    build_aa_prompt,
    parse_aa_output,
    extract_answer_from_completion,
)
from src.agents.memory_manager import (
    build_mm_prompt,
    parse_mm_output,
    execute_mm_operation,
)


# ============================================================
# Phase 1 — Scoring (src/common/scoring.py)
# ============================================================

class TestNormalizeAnswer:
    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("a cat an egg the dog") == "cat egg dog"

    def test_whitespace_collapse(self):
        assert normalize_answer("  hello   world  ") == "hello world"

    def test_combined(self):
        assert normalize_answer("The Quick, Brown Fox!") == "quick brown fox"


class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("the cat sat", "the cat sat") == 1.0

    def test_no_overlap(self):
        assert token_f1("apple", "banana") == 0.0

    def test_partial_overlap(self):
        # pred="cat sat" ref="cat sat mat" → normalized same
        # pred tokens: [cat, sat], ref tokens: [cat, sat, mat]
        # common=2, precision=2/2=1.0, recall=2/3, f1=2*(1*2/3)/(1+2/3)=0.8
        assert abs(token_f1("cat sat", "cat sat mat") - 0.8) < 1e-6

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

    def test_pred_empty_ref_nonempty(self):
        assert token_f1("", "something") == 0.0

    def test_ref_empty_pred_nonempty(self):
        assert token_f1("something", "") == 0.0

    def test_articles_stripped(self):
        # "The Buddy" normalizes to "buddy", "Buddy" normalizes to "buddy"
        assert token_f1("The Buddy", "Buddy") == 1.0


class TestBleu1:
    def test_exact_match(self):
        assert bleu1("cat sat mat", "cat sat mat") == 1.0

    def test_empty_prediction(self):
        assert bleu1("", "something") == 0.0

    def test_partial_overlap(self):
        # pred="cat dog" ref="cat bird" → clipped=1, precision=1/2=0.5
        # bp = min(1.0, 2/2) = 1.0 → 0.5
        assert abs(bleu1("cat dog", "cat bird") - 0.5) < 1e-6

    def test_brevity_penalty(self):
        # Short prediction vs long reference triggers brevity penalty
        # pred="cat" → ["cat"], ref="cat sat on the mat" → ["cat", "sat", "on", "mat"]
        # (articles stripped: "the" removed)
        # clipped=1, precision=1/1=1.0, bp=min(1.0, 1/4)=0.25 → 0.25
        assert abs(bleu1("cat", "cat sat on the mat") - 0.25) < 1e-6

    def test_empty_reference(self):
        # ref empty → ref_tokens=[], bp=min(1.0, len_pred/max(0,1))
        # pred="cat" → pred_tokens=["cat"], ref_tokens=[]
        # clipped=0, precision=0/1=0.0 → 0.0
        assert bleu1("cat", "") == 0.0


class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello", "hello") == 1.0

    def test_different(self):
        assert exact_match("hello", "world") == 0.0

    def test_case_insensitive(self):
        assert exact_match("Hello", "hello") == 1.0

    def test_punctuation_ignored(self):
        assert exact_match("hello!", "hello") == 1.0

    def test_articles_ignored(self):
        assert exact_match("The cat", "cat") == 1.0


class TestComputeReward:
    def test_f1_dispatch(self):
        assert compute_reward("cat", "cat", "f1") == 1.0

    def test_bleu1_dispatch(self):
        assert compute_reward("cat", "cat", "bleu1") == 1.0

    def test_exact_match_dispatch(self):
        assert compute_reward("cat", "cat", "exact_match") == 1.0

    def test_combined_weights(self):
        # For exact match: f1=1.0, bleu1=1.0, em=1.0 → 0.7+0.2+0.1=1.0
        assert abs(compute_reward("cat", "cat", "combined") - 1.0) < 1e-9

    def test_combined_partial(self):
        # pred="cat dog" ref="cat bird"
        f1_val = token_f1("cat dog", "cat bird")
        bleu_val = bleu1("cat dog", "cat bird")
        em_val = exact_match("cat dog", "cat bird")
        expected = 0.7 * f1_val + 0.2 * bleu_val + 0.1 * em_val
        assert abs(compute_reward("cat dog", "cat bird", "combined") - expected) < 1e-9

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown reward type"):
            compute_reward("a", "b", "nonexistent")


# ============================================================
# Phase 2 — MemoryEntry (src/memory/entry.py)
# ============================================================

class TestMemoryEntry:
    def test_creation_defaults(self):
        e = MemoryEntry(entry_id="abc", content="hello", source_session=1)
        assert e.entry_id == "abc"
        assert e.content == "hello"
        assert e.source_session == 1
        assert e.timestamp is None
        assert e.created_at == 0
        assert e.updated_at == 0

    def test_creation_with_optionals(self):
        e = MemoryEntry(
            entry_id="def", content="world", source_session=2,
            timestamp="2025-01-01", created_at=5, updated_at=7,
        )
        assert e.timestamp == "2025-01-01"
        assert e.created_at == 5
        assert e.updated_at == 7

    def test_to_dict_roundtrip(self):
        e = MemoryEntry(
            entry_id="abc", content="hello", source_session=1,
            timestamp="2025-06-01", created_at=3, updated_at=4,
        )
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        assert e == e2

    def test_to_dict_keys(self):
        e = MemoryEntry(entry_id="x", content="y", source_session=0)
        d = e.to_dict()
        assert set(d.keys()) == {
            "entry_id", "content", "source_session",
            "timestamp", "created_at", "updated_at",
        }

    def test_str(self):
        e = MemoryEntry(entry_id="abc", content="hello world", source_session=1)
        assert str(e) == "[abc] hello world"

    def test_from_dict_extra_keys_raises(self):
        with pytest.raises(TypeError):
            MemoryEntry.from_dict({
                "entry_id": "x", "content": "y", "source_session": 0,
                "bogus_key": 42,
            })


# ============================================================
# Phase 3 — MemoryBank (src/memory/bank.py)
# ============================================================

class TestMemoryBank:
    def test_empty_bank(self):
        bank = MemoryBank()
        assert bank.size() == 0
        assert bank.get_all() == []

    def test_add(self):
        bank = MemoryBank()
        eid = bank.add("Alice likes cats", source_session=1)
        assert isinstance(eid, str) and len(eid) == 8
        assert bank.size() == 1
        entry = bank.get_by_id(eid)
        assert entry is not None
        assert entry.content == "Alice likes cats"
        assert entry.source_session == 1

    def test_add_multiple(self):
        bank = MemoryBank()
        id1 = bank.add("fact one", source_session=1)
        id2 = bank.add("fact two", source_session=1)
        assert bank.size() == 2
        assert id1 != id2

    def test_update_existing(self):
        bank = MemoryBank()
        eid = bank.add("Alice likes cats", source_session=1)
        bank.advance_turn()
        result = bank.update(eid, "Alice likes dogs")
        assert result is True
        assert bank.get_by_id(eid).content == "Alice likes dogs"
        assert bank.get_by_id(eid).updated_at == 1  # after advance_turn

    def test_update_missing(self):
        bank = MemoryBank()
        assert bank.update("nonexistent", "new content") is False

    def test_delete_existing(self):
        bank = MemoryBank()
        eid = bank.add("temp fact", source_session=1)
        assert bank.size() == 1
        result = bank.delete(eid)
        assert result is True
        assert bank.size() == 0
        assert bank.get_by_id(eid) is None

    def test_delete_missing(self):
        bank = MemoryBank()
        assert bank.delete("nonexistent") is False

    def test_noop(self):
        bank = MemoryBank()
        bank.add("fact", source_session=1)
        bank.noop()
        assert bank.size() == 1  # unchanged

    def test_get_all(self):
        bank = MemoryBank()
        bank.add("one", source_session=1)
        bank.add("two", source_session=2)
        entries = bank.get_all()
        assert len(entries) == 2
        contents = {e.content for e in entries}
        assert contents == {"one", "two"}

    def test_get_by_id_missing(self):
        bank = MemoryBank()
        assert bank.get_by_id("nope") is None

    def test_advance_turn(self):
        bank = MemoryBank()
        id1 = bank.add("turn0", source_session=1)
        bank.advance_turn()
        id2 = bank.add("turn1", source_session=1)
        # Same content would give same ID without the turn counter
        # Different IDs because _turn_counter differs
        assert bank.get_by_id(id1).created_at == 0
        assert bank.get_by_id(id2).created_at == 1

    def test_search_keyword(self):
        bank = MemoryBank()
        bank.add("Alice likes cats", source_session=1)
        bank.add("Bob enjoys running", source_session=1)
        bank.add("Alice enjoys swimming", source_session=1)
        results = bank.search_keyword("Alice cats")
        assert len(results) >= 1
        # "Alice likes cats" should be top (2 word overlap)
        assert results[0].content == "Alice likes cats"

    def test_search_keyword_empty_query(self):
        bank = MemoryBank()
        bank.add("something", source_session=1)
        # Empty query has no terms, so no overlap
        results = bank.search_keyword("")
        assert results == []

    def test_search_keyword_top_k(self):
        bank = MemoryBank()
        for i in range(10):
            bank.add(f"word match item {i}", source_session=1)
        results = bank.search_keyword("word", top_k=3)
        assert len(results) == 3

    def test_json_roundtrip(self):
        bank = MemoryBank()
        bank.add("fact one", source_session=1, timestamp="2025-01-01")
        bank.advance_turn()
        bank.add("fact two", source_session=2)
        json_str = bank.to_json()
        bank2 = MemoryBank.from_json(json_str)
        assert bank2.size() == 2
        contents = {e.content for e in bank2.get_all()}
        assert contents == {"fact one", "fact two"}
        # Verify timestamp preserved
        for e in bank2.get_all():
            if e.content == "fact one":
                assert e.timestamp == "2025-01-01"

    def test_json_empty_bank(self):
        bank = MemoryBank()
        json_str = bank.to_json()
        bank2 = MemoryBank.from_json(json_str)
        assert bank2.size() == 0

    def test_format_for_prompt_with_entries(self):
        bank = MemoryBank()
        bank.add("Alice is 25", source_session=1, timestamp="Jan 2025")
        formatted = bank.format_for_prompt()
        assert "Alice is 25" in formatted
        assert "(session 1, Jan 2025)" in formatted

    def test_format_for_prompt_empty(self):
        bank = MemoryBank()
        assert bank.format_for_prompt() == "No memories stored."

    def test_format_for_prompt_no_timestamp(self):
        bank = MemoryBank()
        bank.add("Bob likes tea", source_session=3)
        formatted = bank.format_for_prompt()
        assert "(session 3)" in formatted
        # Should NOT contain a comma after session number when no timestamp
        # Pattern: "(session 3)" not "(session 3, )"
        assert "(session 3," not in formatted


# ============================================================
# Phase 4 — Answer Agent (src/agents/answer_agent.py)
# ============================================================

class TestBuildAAPrompt:
    def test_with_memories(self):
        memories = [
            MemoryEntry(entry_id="m1", content="Alice is 25", source_session=1),
            MemoryEntry(entry_id="m2", content="Bob likes tea", source_session=2),
        ]
        msgs = build_aa_prompt("How old is Alice?", memories)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "How old is Alice?" in msgs[1]["content"]
        assert "Alice is 25" in msgs[1]["content"]
        assert "top 2" in msgs[1]["content"]

    def test_empty_memories(self):
        msgs = build_aa_prompt("What is the answer?", [])
        assert "No relevant memories found." in msgs[1]["content"]
        assert "top 0" in msgs[1]["content"]


class TestParseAAOutput:
    def test_valid_xml(self):
        raw = """
<selected_memories>
m1, m2
</selected_memories>
<reasoning>
Alice is 25 based on memory m1.
</reasoning>
<answer>
Alice is 25 years old.
</answer>
"""
        result = parse_aa_output(raw)
        assert result["selected_memories"] == ["m1", "m2"]
        assert "Alice is 25" in result["reasoning"]
        assert result["answer"] == "Alice is 25 years old."

    def test_missing_answer_tag_falls_back_to_last_line(self):
        raw = """
<selected_memories>m1</selected_memories>
<reasoning>Some reasoning here.</reasoning>
The answer is 42.
"""
        result = parse_aa_output(raw)
        assert result["answer"] == "The answer is 42."

    def test_missing_all_tags(self):
        raw = "Just a plain text answer."
        result = parse_aa_output(raw)
        assert result["selected_memories"] == []
        assert result["reasoning"] == ""
        assert result["answer"] == "Just a plain text answer."

    def test_empty_string(self):
        result = parse_aa_output("")
        assert result["answer"] == ""
        assert result["selected_memories"] == []
        assert result["reasoning"] == ""


class TestExtractAnswerFromCompletion:
    def test_xml_extraction(self):
        text = "blah blah <answer>Paris</answer> more text"
        assert extract_answer_from_completion(text) == "Paris"

    def test_last_line_fallback(self):
        text = "Some reasoning\nThe capital is Paris"
        assert extract_answer_from_completion(text) == "The capital is Paris"

    def test_empty_string(self):
        assert extract_answer_from_completion("") == ""

    def test_whitespace_only(self):
        assert extract_answer_from_completion("   \n  \n  ") == ""


# ============================================================
# Phase 5 — Memory Manager (src/agents/memory_manager.py)
# ============================================================

class TestBuildMMPrompt:
    def test_empty_bank(self):
        bank = MemoryBank()
        msgs = build_mm_prompt(bank, session_id=1, turn_id=0,
                               speaker="User", message="Hi there")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "No memories stored." in msgs[1]["content"]
        assert "Hi there" in msgs[1]["content"]
        assert "Session 1" in msgs[1]["content"]

    def test_with_memories(self):
        bank = MemoryBank()
        bank.add("Alice is 25", source_session=1)
        msgs = build_mm_prompt(bank, session_id=1, turn_id=1,
                               speaker="User", message="How old?")
        assert "Alice is 25" in msgs[1]["content"]


class TestParseMMOutput:
    def test_valid_add(self):
        raw = '{"op": "ADD", "content": "Alice is 25"}'
        result = parse_mm_output(raw)
        assert result["op"] == "ADD"
        assert result["content"] == "Alice is 25"

    def test_valid_update(self):
        raw = '{"op": "UPDATE", "entry_id": "abc123", "content": "Alice is 26"}'
        result = parse_mm_output(raw)
        assert result["op"] == "UPDATE"
        assert result["entry_id"] == "abc123"

    def test_valid_delete(self):
        raw = '{"op": "DELETE", "entry_id": "abc123"}'
        result = parse_mm_output(raw)
        assert result["op"] == "DELETE"

    def test_valid_noop(self):
        raw = '{"op": "NOOP"}'
        result = parse_mm_output(raw)
        assert result["op"] == "NOOP"

    def test_markdown_fences(self):
        raw = '```json\n{"op": "ADD", "content": "fact"}\n```'
        result = parse_mm_output(raw)
        assert result["op"] == "ADD"

    def test_extra_text_around_json(self):
        raw = 'Based on the turn, I think:\n{"op": "ADD", "content": "new fact"}\nDone.'
        result = parse_mm_output(raw)
        assert result["op"] == "ADD"
        assert result["content"] == "new fact"

    def test_invalid_json(self):
        raw = "This is not JSON at all."
        result = parse_mm_output(raw)
        assert result["op"] == "NOOP"

    def test_unknown_op(self):
        raw = '{"op": "MERGE", "content": "something"}'
        result = parse_mm_output(raw)
        assert result["op"] == "NOOP"

    def test_empty_string(self):
        result = parse_mm_output("")
        assert result["op"] == "NOOP"


class TestExecuteMMOperation:
    def test_add(self):
        bank = MemoryBank()
        status = execute_mm_operation(
            {"op": "ADD", "content": "Alice is 25"}, bank, session_id=1,
        )
        assert "ADD:" in status
        assert bank.size() == 1

    def test_add_empty_content(self):
        bank = MemoryBank()
        status = execute_mm_operation(
            {"op": "ADD", "content": ""}, bank, session_id=1,
        )
        assert "skipped" in status
        assert bank.size() == 0

    def test_update_existing(self):
        bank = MemoryBank()
        eid = bank.add("old fact", source_session=1)
        status = execute_mm_operation(
            {"op": "UPDATE", "entry_id": eid, "content": "new fact"}, bank, session_id=1,
        )
        assert "ok" in status
        assert bank.get_by_id(eid).content == "new fact"

    def test_update_nonexistent(self):
        bank = MemoryBank()
        status = execute_mm_operation(
            {"op": "UPDATE", "entry_id": "nope", "content": "new"}, bank, session_id=1,
        )
        assert "failed" in status

    def test_update_missing_id(self):
        bank = MemoryBank()
        status = execute_mm_operation(
            {"op": "UPDATE", "content": "new"}, bank, session_id=1,
        )
        assert "skipped" in status

    def test_delete_existing(self):
        bank = MemoryBank()
        eid = bank.add("temp", source_session=1)
        status = execute_mm_operation(
            {"op": "DELETE", "entry_id": eid}, bank, session_id=1,
        )
        assert "ok" in status
        assert bank.size() == 0

    def test_delete_nonexistent(self):
        bank = MemoryBank()
        status = execute_mm_operation(
            {"op": "DELETE", "entry_id": "nope"}, bank, session_id=1,
        )
        assert "failed" in status

    def test_delete_missing_id(self):
        bank = MemoryBank()
        status = execute_mm_operation(
            {"op": "DELETE"}, bank, session_id=1,
        )
        assert "skipped" in status

    def test_noop(self):
        bank = MemoryBank()
        bank.add("existing", source_session=1)
        status = execute_mm_operation({"op": "NOOP"}, bank, session_id=1)
        assert status == "NOOP"
        assert bank.size() == 1


# ============================================================
# Phase 6 — Heuristic Memory (src/memory/heuristic.py)
# ============================================================

class TestSkipWords:
    def test_expected_words_present(self):
        for word in ("hi", "hello", "hey", "thanks", "bye", "ok", "okay", "yes", "no"):
            assert word in SKIP_WORDS

    def test_is_set(self):
        assert isinstance(SKIP_WORDS, set)


class TestBuildHeuristicMemory:
    def _make_example(self, turns):
        """Helper to build an example dict with one session."""
        return {
            "sessions": [{
                "session_id": 1,
                "date_time": "Jan 2025",
                "turns": turns,
            }],
        }

    def test_basic(self):
        example = self._make_example([
            {"speaker": "User", "text": "I really love exploring new cities and traveling abroad"},
        ])
        mems = build_heuristic_memory(example)
        assert len(mems) == 1
        assert "User:" in mems[0]
        assert "traveling abroad" in mems[0]
        assert "(session 1, Jan 2025)" in mems[0]

    def test_skips_short_messages(self):
        example = self._make_example([
            {"speaker": "User", "text": "Oh ok sure"},  # <=5 words
        ])
        mems = build_heuristic_memory(example)
        assert len(mems) == 0

    def test_skips_skip_word_starts(self):
        example = self._make_example([
            {"speaker": "User", "text": "hi there how are you doing today friend"},
        ])
        mems = build_heuristic_memory(example)
        assert len(mems) == 0

    def test_truncates_long_text(self):
        long_text = "word " * 200  # 1000 chars
        example = self._make_example([
            {"speaker": "User", "text": long_text},
        ])
        mems = build_heuristic_memory(example)
        assert len(mems) == 1
        # Content (after "User: ") should be at most 300 chars of the original text
        content_part = mems[0].split("User: ")[1].split(" (session")[0]
        assert len(content_part) <= 300

    def test_no_sessions(self):
        assert build_heuristic_memory({}) == []
        assert build_heuristic_memory({"sessions": []}) == []

    def test_multiple_sessions(self):
        example = {
            "sessions": [
                {
                    "session_id": 1, "date_time": "Jan",
                    "turns": [{"speaker": "User", "text": "I love cats and dogs very much indeed"}],
                },
                {
                    "session_id": 2, "date_time": "Feb",
                    "turns": [{"speaker": "Bot", "text": "The weather will be sunny and warm tomorrow morning"}],
                },
            ],
        }
        mems = build_heuristic_memory(example)
        assert len(mems) == 2
        assert "(session 1, Jan)" in mems[0]
        assert "(session 2, Feb)" in mems[1]


class TestBuildHeuristicMemories:
    def _make_sessions(self, n_turns_per_session=3, n_sessions=2):
        sessions = []
        for s in range(n_sessions):
            turns = []
            for t in range(n_turns_per_session):
                turns.append({
                    "speaker": "User",
                    "text": f"This is a longer message about topic {s}_{t} for testing purposes",
                })
            sessions.append({
                "session_id": s,
                "date_time": f"Day {s}",
                "turns": turns,
            })
        return sessions

    def test_basic(self):
        sessions = self._make_sessions()
        mems = build_heuristic_memories(sessions)
        assert len(mems) == 6  # 2 sessions × 3 turns

    def test_empty_sessions(self):
        assert build_heuristic_memories([]) == []

    def test_max_memories_cap(self):
        sessions = self._make_sessions(n_turns_per_session=10, n_sessions=5)
        mems = build_heuristic_memories(sessions, max_memories=5)
        assert len(mems) == 5

    def test_max_memories_no_cap_when_under(self):
        sessions = self._make_sessions(n_turns_per_session=2, n_sessions=1)
        mems = build_heuristic_memories(sessions, max_memories=200)
        assert len(mems) == 2


# ============================================================
# Phase 7 — Config (src/common/config.py)
# ============================================================

class TestLoadConfig:
    def test_load_valid_yaml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8",
        ) as f:
            f.write("model_name: qwen\nbatch_size: 4\n")
            f.flush()
            cfg = load_config(f.name)
        assert cfg == {"model_name": "qwen", "batch_size": 4}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


# ============================================================
# Phase 7b — Prompt Templates (src/common/prompts.py)
# ============================================================

class TestPromptTemplates:
    def test_aa_user_template_placeholders(self):
        assert "{num_retrieved}" in AA_USER_TEMPLATE
        assert "{memories}" in AA_USER_TEMPLATE
        assert "{question}" in AA_USER_TEMPLATE

    def test_mm_user_template_placeholders(self):
        assert "{memories}" in MM_USER_TEMPLATE
        assert "{session_id}" in MM_USER_TEMPLATE
        assert "{turn_id}" in MM_USER_TEMPLATE
        assert "{speaker}" in MM_USER_TEMPLATE
        assert "{message}" in MM_USER_TEMPLATE

    def test_aa_system_prompt_nonempty(self):
        assert len(AA_SYSTEM_PROMPT) > 50

    def test_mm_system_prompt_nonempty(self):
        assert len(MM_SYSTEM_PROMPT) > 50
