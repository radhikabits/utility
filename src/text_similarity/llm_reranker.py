from __future__ import annotations

from typing import List, Optional, Protocol, Tuple
import math
import pandas as pd


class SimilarityScorer(Protocol):
    def score_similarity(self, source_text: str, target_text: str) -> float:
        ...


def rerank_with_llm(
    from_texts: List[str],
    to_texts: List[str],
    prefilter_matrix: pd.DataFrame,
    scorer: SimilarityScorer,
    top_k: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Rerank candidates with an LLM-like scorer using a prefilter matrix.

    Returns (best_indices, best_scores) as pandas Series of length len(from_texts).
    """
    n_from = len(from_texts)
    n_to = len(to_texts)

    best_indices: List[int] = []
    best_scores: List[float] = []

    for i in range(n_from):
        pre_scores = prefilter_matrix.iloc[i].to_numpy()
        k = max(1, min(top_k, n_to))
        top_idx = pre_scores.argsort()[::-1][:k]

        src = str(from_texts[i] if from_texts[i] is not None else "")
        best_j = int(top_idx[0])
        best_score = -math.inf

        for j in top_idx:
            tgt = str(to_texts[int(j)] if to_texts[int(j)] is not None else "")
            try:
                score = float(scorer.score_similarity(src, tgt))
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_j = int(j)

        best_indices.append(best_j)
        best_scores.append(max(0.0, min(1.0, float(best_score))))

    return pd.Series(best_indices), pd.Series(best_scores)
