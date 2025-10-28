from __future__ import annotations

from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_matrix(from_texts: List[str], to_texts: List[str]) -> pd.DataFrame:
    """Compute TF-IDF cosine similarity matrix between from_texts (rows) and to_texts (cols).

    Returns a pandas DataFrame of shape (len(from_texts), len(to_texts)).
    """
    # Normalize to strings and handle None
    to_texts = [str(t) if t is not None else "" for t in to_texts]
    from_texts = [str(t) if t is not None else "" for t in from_texts]

    vectorizer = TfidfVectorizer(stop_words="english")
    corpus = list(to_texts) + list(from_texts)
    X = vectorizer.fit_transform(corpus)
    G = X[: len(to_texts)]
    A = X[len(to_texts) :]
    sims = cosine_similarity(A, G)
    return pd.DataFrame(sims)
