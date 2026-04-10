"""
MemoryEntry dataclass — the atomic unit stored in a MemoryBank.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class MemoryEntry:
    """A single memory entry in the bank."""
    entry_id: str
    content: str
    source_session: int
    timestamp: Optional[str] = None
    created_at: int = 0   # turn number when created
    updated_at: int = 0   # turn number when last updated

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "source_session": self.source_session,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        return cls(**d)

    def __str__(self) -> str:
        return f"[{self.entry_id}] {self.content}"
