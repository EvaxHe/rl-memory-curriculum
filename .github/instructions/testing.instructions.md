---
description: "Use when writing or modifying tests, adding test coverage, or debugging test failures."
applyTo: "tests/**"
---
# Testing Conventions

## Test Types

| File | Purpose | Dependencies |
|------|---------|-------------|
| `test_utilities.py` | Unit tests for pure utilities | None — no ML, no data files, no GPU |
| `test_dry_run.py` | Integration tests (pipeline) | Needs `data/processed/` JSONL files |
| `test_eval_dry_run.py` | Eval pipeline with mocks | Needs `data/processed/` JSONL files, uses `unittest.mock` |

## Rules

- Use **pytest** style (plain `assert`, `pytest.raises`). No unittest.TestCase.
- Unit tests (`test_utilities.py`) must be fully self-contained: no data files, no ML model loading, no GPU.
- Use `tempfile` for any file I/O in tests. Never write to project directories.
- Always pass `encoding="utf-8"` when reading/writing files.
- Group related tests in classes (`class TestMemoryBank:`). Use descriptive method names.
- Run with: `uv run python -m pytest tests/ -v`

## Patterns

```python
# Pure utility test — no fixtures needed
class TestTokenF1:
    def test_exact_match(self):
        assert token_f1("cat sat", "cat sat") == 1.0

    def test_both_empty(self):
        assert token_f1("", "") == 1.0

# Integration test with mocking
from unittest.mock import MagicMock, patch

def test_eval_with_mocked_vllm():
    with patch("src.eval.inference.LLM") as mock_llm:
        mock_llm.return_value.generate.return_value = [...]
```

## Floating Point

Use `abs(result - expected) < 1e-9` for floating-point comparisons, not `==`.
