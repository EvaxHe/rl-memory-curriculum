"""
Embedding-based retrieval for the memory bank.

Memory-R1 retrieves 60 candidates via RAG before the Answer Agent
filters them down. We use sentence-transformers + FAISS for this.

For budget reasons, we use a small but effective model:
- all-MiniLM-L6-v2 (22M params, 384 dims) — fast, good quality
- FAISS IndexFlatIP for exact inner product search (small bank, no need for ANN)
"""
import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded globals to avoid import overhead when not needed
_model = None
_tokenizer = None


def get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Falling back to keyword search. "
                "Install with: uv add sentence-transformers"
            )
            return None
    return _model


def embed_texts(texts: list[str],
                model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
                ) -> Optional[np.ndarray]:
    """Embed a list of texts. Returns (N, D) array or None if unavailable."""
    model = get_embedder(model_name)
    if model is None:
        return None
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS index for inner product search."""
    try:
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        index.add(embeddings)
        return index
    except ImportError:
        logger.warning("faiss-cpu not installed. Using numpy fallback.")
        return None


def search_faiss(query_embedding: np.ndarray, index, top_k: int = 60):
    """Search FAISS index. Returns (scores, indices)."""
    if index is None:
        return None, None
    scores, indices = index.search(query_embedding.reshape(1, -1), top_k)
    return scores[0], indices[0]


def search_numpy_fallback(query_embedding: np.ndarray,
                          corpus_embeddings: np.ndarray,
                          top_k: int = 60):
    """Numpy fallback when FAISS is not available."""
    scores = corpus_embeddings @ query_embedding.T
    scores = scores.flatten()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return scores[top_indices], top_indices
