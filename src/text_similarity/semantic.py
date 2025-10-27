from __future__ import annotations

from typing import List
import pandas as pd


def compute_semantic_matrix(from_texts: List[str], to_texts: List[str]) -> pd.DataFrame:
    """Compute semantic cosine similarity matrix using sentence-transformers.

    Lazily imports heavy deps to keep optional when unused.
    Returns a pandas DataFrame of shape (len(from_texts), len(to_texts)).
    """
    # Normalize to strings
    to_texts = [str(t) if t is not None else "" for t in to_texts]
    from_texts = [str(t) if t is not None else "" for t in from_texts]

    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "sentence-transformers is required for semantic mode. Install it or choose a different mode."
        ) from exc

    model = SentenceTransformer("all-MiniLM-L6-v2")
    to_embeddings = model.encode(to_texts, convert_to_tensor=True)
    from_embeddings = model.encode(from_texts, convert_to_tensor=True)
    sims = util.cos_sim(from_embeddings, to_embeddings)
    return pd.DataFrame(sims.cpu().numpy())
