"""
MemoryBank — dict-backed storage with CRUD operations, state, and serialization.

Search, formatting, and retrieval logic live in sibling modules
(search.py, formatting.py, retriever.py) to keep this class focused on storage.
"""
import json
import hashlib
from typing import Optional

from src.memory.entry import MemoryEntry


class MemoryBank:
    """
    Dict-based memory bank supporting ADD, UPDATE, DELETE, NOOP.
    """

    def __init__(self, use_embeddings: bool = True):
        self.entries: dict[str, MemoryEntry] = {}
        self._turn_counter = 0
        self._use_embeddings = use_embeddings
        self._embedder = None
        self._index = None

    def _generate_id(self, content: str) -> str:
        raw = f"{content}_{self._turn_counter}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    # ---- CRUD Operations (called by Memory Manager) ----

    def add(self, content: str, source_session: int,
            timestamp: Optional[str] = None) -> str:
        entry_id = self._generate_id(content)
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            source_session=source_session,
            timestamp=timestamp,
            created_at=self._turn_counter,
            updated_at=self._turn_counter,
        )
        self.entries[entry_id] = entry
        return entry_id

    def update(self, entry_id: str, new_content: str) -> bool:
        if entry_id not in self.entries:
            return False
        self.entries[entry_id].content = new_content
        self.entries[entry_id].updated_at = self._turn_counter
        return True

    def delete(self, entry_id: str) -> bool:
        if entry_id not in self.entries:
            return False
        del self.entries[entry_id]
        return True

    def noop(self) -> None:
        """Explicit no-operation."""
        pass

    # ---- Retrieval ----

    def get_all(self) -> list[MemoryEntry]:
        return list(self.entries.values())

    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        return self.entries.get(entry_id)

    def search_keyword(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """Keyword overlap search (fallback)."""
        query_terms = set(query.lower().split())
        scored = []
        for entry in self.entries.values():
            content_terms = set(entry.content.lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def search(self, query: str, top_k: int = 60) -> list[MemoryEntry]:
        """
        Retrieve top-k memories. Uses embedding search if available,
        otherwise keyword overlap. Memory-R1 retrieves 60 candidates.
        """
        if not self.entries:
            return []

        # Try embedding-based search first
        try:
            from src.memory.retriever import embed_texts, search_numpy_fallback
            entries_list = list(self.entries.values())
            texts = [e.content for e in entries_list]
            corpus_emb = embed_texts(texts)
            if corpus_emb is not None:
                query_emb = embed_texts([query])
                if query_emb is not None:
                    scores, indices = search_numpy_fallback(
                        query_emb[0], corpus_emb, top_k=min(top_k, len(entries_list))
                    )
                    return [entries_list[i] for i in indices if i < len(entries_list)]
        except Exception:
            pass

        # Fallback to keyword search
        return self.search_keyword(query, top_k=top_k)

    # ---- State management ----

    def advance_turn(self):
        self._turn_counter += 1

    def size(self) -> int:
        return len(self.entries)

    def to_json(self) -> str:
        return json.dumps([e.to_dict() for e in self.entries.values()], indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "MemoryBank":
        bank = cls()
        for d in json.loads(json_str):
            entry = MemoryEntry.from_dict(d)
            bank.entries[entry.entry_id] = entry
        return bank

    def format_for_prompt(self, entries: Optional[list[MemoryEntry]] = None) -> str:
        """Format memory entries for inclusion in LLM prompt."""
        entries = entries or self.get_all()
        if not entries:
            return "No memories stored."
        lines = []
        for e in entries:
            meta = f"(session {e.source_session}"
            if e.timestamp:
                meta += f", {e.timestamp}"
            meta += ")"
            lines.append(f"- [{e.entry_id}] {e.content} {meta}")
        return "\n".join(lines)
