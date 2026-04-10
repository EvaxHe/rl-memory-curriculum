"""
Heuristic memory builder — constructs memory entries from dialogue sessions
without using a trained Memory Manager model.

Used for AA training (before MM is trained) and as a baseline in evaluation.
"""

SKIP_WORDS = {"hi", "hello", "hey", "thanks", "bye", "ok", "okay", "yes", "no"}


def build_heuristic_memory(example: dict) -> list[str]:
    """
    Build memory entries from a single example using simple heuristics (no RL).
    Used to create AA training prompts before MM is trained.

    Args:
        example: A training/test example dict with "sessions" key.

    Returns:
        List of memory strings.
    """
    memories = []

    for session in example.get("sessions", []):
        sid = session.get("session_id", 0)
        dt = session.get("date_time", "")
        for turn in session.get("turns", []):
            text = turn.get("text", "").strip()
            speaker = turn.get("speaker", "")
            words = text.lower().split()
            if len(words) > 5 and not any(w in SKIP_WORDS for w in words[:2]):
                mem = f"{speaker}: {text[:300]}"
                if dt:
                    mem += f" (session {sid}, {dt})"
                memories.append(mem)

    return memories


def build_heuristic_memories(sessions, max_memories=200):
    """
    Build memory entries from sessions using heuristics.
    Variant that takes sessions directly (used by eval).

    Args:
        sessions: List of session dicts with "turns" key.
        max_memories: Maximum number of memories to return.

    Returns:
        List of memory strings.
    """
    memories = []
    for session in sessions:
        sid = session.get("session_id", 0)
        dt = session.get("date_time", "")
        for turn in session.get("turns", []):
            text = turn.get("text", "").strip()
            speaker = turn.get("speaker", "")
            words = text.lower().split()
            if len(words) > 5 and not any(w in SKIP_WORDS for w in words[:2]):
                mem = f"{speaker}: {text[:300]}"
                if dt:
                    mem += f" (session {sid}, {dt})"
                memories.append(mem)
    if len(memories) > max_memories:
        step = len(memories) / max_memories
        memories = [memories[int(i * step)] for i in range(max_memories)]
    return memories


def retrieve_memories(question: str, all_memories: list[str],
                      top_k: int = 20) -> list[str]:
    """
    Retrieve top-k memories for a question. Embedding search with keyword fallback.

    Shared by both training (AA prompt construction) and evaluation.
    """
    if not all_memories:
        return []

    # Try embedding retrieval first
    try:
        from src.memory.retriever import embed_texts, search_numpy_fallback
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
    scored = [(len(q_words & set(m.lower().split())), m) for m in all_memories]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]
