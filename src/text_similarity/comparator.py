from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, List, Dict, Tuple
import logging
import math

import pandas as pd

# Delegate algorithm implementations to dedicated modules
from .tfidf import compute_tfidf_matrix
from .semantic import compute_semantic_matrix
from .llm_reranker import rerank_with_llm

# Avoid importing sentence-transformers at module import to keep optional when unused


logger = logging.getLogger(__name__)


Mode = Literal["tfidf", "semantic", "llm", "both", "all"]


@dataclass
class TextSimilarityComparator:
    """
    Generic comparator to evaluate similarity between two text datasets.

    Supports:
    - TF-IDF cosine similarity
    - Semantic similarity (sentence-transformers)
    - LLM-based reranking/scoring over a top-k candidate set
    - Combined weighted results (average by default)
    """

    from_csv_path: str
    to_csv_path: str
    from_text_column: str
    to_text_column: str
    from_additional_columns: Optional[List[str]] = None
    to_additional_columns: Optional[List[str]] = None
    mode: Mode = "both"

    # Combination weights (used when multiple metrics are present)
    weight_tfidf: float = 1.0
    weight_semantic: float = 1.0
    weight_llm: float = 1.0

    # LLM settings
    llm_client: Optional[object] = None  # Protocol defined in llm.py
    llm_top_k: int = 5
    llm_prefilter: Literal["tfidf", "semantic"] = "tfidf"

    # Internal state (populated during run)
    from_df: Optional[pd.DataFrame] = field(init=False, default=None)
    to_df: Optional[pd.DataFrame] = field(init=False, default=None)

    tfidf_matrix: Optional["pd.DataFrame"] = field(init=False, default=None)
    semantic_matrix: Optional["pd.DataFrame"] = field(init=False, default=None)

    # LLM results stored as per-row best
    llm_best_idx: Optional[pd.Series] = field(init=False, default=None)
    llm_scores: Optional[pd.Series] = field(init=False, default=None)

    results_df: Optional[pd.DataFrame] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()  # normalize
        self.from_additional_columns = self.from_additional_columns or []
        self.to_additional_columns = self.to_additional_columns or []

    # ------------------------------------------------------------------
    def load_data(self) -> None:
        """Load both datasets from CSV."""
        self.from_df = pd.read_csv(self.from_csv_path)
        self.to_df = pd.read_csv(self.to_csv_path)

        # Validate columns
        for name, df, col in (
            ("from_text_column", self.from_df, self.from_text_column),
            ("to_text_column", self.to_df, self.to_text_column),
        ):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' (for {name}) not found in CSV")
        for col in self.from_additional_columns:
            if col not in self.from_df.columns:
                raise ValueError(f"Additional column '{col}' not found in from_csv")
        for col in self.to_additional_columns:
            if col not in self.to_df.columns:
                raise ValueError(f"Additional column '{col}' not found in to_csv")

        # Normalize NaNs
        self.from_df[self.from_text_column] = (
            self.from_df[self.from_text_column].astype(str).fillna("")
        )
        self.to_df[self.to_text_column] = self.to_df[self.to_text_column].astype(str).fillna("")

    # ------------------------------------------------------------------
    def compute_tfidf_similarity(self) -> None:
        """Compute cosine similarity using TF-IDF."""
        assert self.from_df is not None and self.to_df is not None
        self.tfidf_matrix = compute_tfidf_matrix(
            from_texts=self.from_df[self.from_text_column].tolist(),
            to_texts=self.to_df[self.to_text_column].tolist(),
        )

    # ------------------------------------------------------------------
    def compute_semantic_similarity(self) -> None:
        """Compute cosine similarity using sentence-transformers embeddings."""
        assert self.from_df is not None and self.to_df is not None
        self.semantic_matrix = compute_semantic_matrix(
            from_texts=self.from_df[self.from_text_column].tolist(),
            to_texts=self.to_df[self.to_text_column].tolist(),
        )

    # ------------------------------------------------------------------
    def compute_llm_similarity(self) -> None:
        """Compute best matches and scores using an LLM scoring client.

        For each 'from' row, we consider top-k candidates chosen by a prefilter
        (TF-IDF by default), then ask the LLM to score each pair, selecting the best.
        """
        assert self.from_df is not None and self.to_df is not None
        if self.llm_client is None:
            raise ValueError("llm_client must be provided for LLM mode")

        # Ensure we have a prefilter matrix
        pre_matrix = None
        if self.llm_prefilter == "semantic" and self.semantic_matrix is not None:
            pre_matrix = self.semantic_matrix
        elif self.tfidf_matrix is not None:
            pre_matrix = self.tfidf_matrix
        else:
            # If no prefilter available, compute TF-IDF quickly
            self.compute_tfidf_similarity()
            pre_matrix = self.tfidf_matrix

        assert pre_matrix is not None

        best_indices, best_scores = rerank_with_llm(
            from_texts=self.from_df[self.from_text_column].tolist(),
            to_texts=self.to_df[self.to_text_column].tolist(),
            prefilter_matrix=pre_matrix,
            scorer=self.llm_client,  # type: ignore[arg-type]
            top_k=self.llm_top_k,
        )

        self.llm_best_idx = best_indices
        self.llm_scores = best_scores

    # ------------------------------------------------------------------
    @staticmethod
    def classify_similarity(score: float) -> str:
        """Label similarity scores into qualitative buckets."""
        if score >= 0.8:
            return "Aligned"
        if score >= 0.6:
            return "Partial"
        return "Misaligned"

    # ------------------------------------------------------------------
    def _best_from_matrix(self, matrix: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        best_idx = matrix.values.argmax(axis=1)
        best_scores = matrix.values.max(axis=1)
        return pd.Series(best_idx), pd.Series(best_scores)

    # ------------------------------------------------------------------
    def generate_results(self) -> None:
        """Generate a unified result dataframe from the computed metrics."""
        assert self.from_df is not None and self.to_df is not None

        # Track which metrics are available
        have_tfidf = self.tfidf_matrix is not None
        have_sem = self.semantic_matrix is not None
        have_llm = self.llm_best_idx is not None and self.llm_scores is not None

        if self.mode in ("tfidf", "both", "all") and not have_tfidf:
            raise ValueError("TF-IDF similarity not computed.")
        if self.mode in ("semantic", "both", "all") and not have_sem:
            raise ValueError("Semantic similarity not computed.")
        if self.mode in ("llm", "all") and not have_llm:
            raise ValueError("LLM similarity not computed.")

        base_data: Dict[str, object] = {
            f"from_{self.from_text_column}": self.from_df[self.from_text_column]
        }
        for col in self.from_additional_columns:
            base_data[f"from_{col}"] = self.from_df[col]

        # Attach metric-wise best matches and scores
        if have_tfidf and self.tfidf_matrix is not None:
            tfidf_best_idx, tfidf_scores = self._best_from_matrix(self.tfidf_matrix)
            base_data[f"to_best_match_tfidf"] = (
                self.to_df[self.to_text_column].iloc[tfidf_best_idx].values
            )
            for col in self.to_additional_columns:
                base_data[f"to_{col}_tfidf"] = self.to_df[col].iloc[tfidf_best_idx].values
            base_data["tfidf_score"] = tfidf_scores

        if have_sem and self.semantic_matrix is not None:
            sem_best_idx, sem_scores = self._best_from_matrix(self.semantic_matrix)
            base_data[f"to_best_match_semantic"] = (
                self.to_df[self.to_text_column].iloc[sem_best_idx].values
            )
            for col in self.to_additional_columns:
                base_data[f"to_{col}_semantic"] = self.to_df[col].iloc[sem_best_idx].values
            base_data["semantic_score"] = sem_scores

        if have_llm and self.llm_best_idx is not None and self.llm_scores is not None:
            base_data[f"to_best_match_llm"] = (
                self.to_df[self.to_text_column].iloc[self.llm_best_idx].values
            )
            for col in self.to_additional_columns:
                base_data[f"to_{col}_llm"] = self.to_df[col].iloc[self.llm_best_idx].values
            base_data["llm_score"] = self.llm_scores

        results = pd.DataFrame(base_data)

        # Combined score
        score_cols: List[Tuple[str, float]] = []
        if "tfidf_score" in results:
            score_cols.append(("tfidf_score", self.weight_tfidf))
        if "semantic_score" in results:
            score_cols.append(("semantic_score", self.weight_semantic))
        if "llm_score" in results:
            score_cols.append(("llm_score", self.weight_llm))

        if score_cols:
            # Weighted average ignoring missing values by re-normalizing weights per row
            def weighted_row_avg(row: pd.Series) -> float:
                total = 0.0
                wsum = 0.0
                for col, w in score_cols:
                    val = row.get(col)
                    if pd.notna(val):
                        total += float(val) * float(w)
                        wsum += float(w)
                return total / wsum if wsum > 0 else float("nan")

            results["combined_score"] = results.apply(weighted_row_avg, axis=1)
            results["comment"] = results["combined_score"].apply(self.classify_similarity)
        else:
            # Fallback to any single score present
            for candidate in ("tfidf_score", "semantic_score", "llm_score"):
                if candidate in results:
                    results["comment"] = results[candidate].apply(self.classify_similarity)
                    break

        self.results_df = results

    # ------------------------------------------------------------------
    def save_results(self, output_path: str = "comparison_results.csv") -> None:
        if self.results_df is None:
            raise ValueError("No results to save. Run generate_results() first.")
        self.results_df.to_csv(output_path, index=False)

    # ------------------------------------------------------------------
    def get_results(self) -> pd.DataFrame:
        if self.results_df is None:
            raise ValueError("No results available. Run generate_results() first.")
        return self.results_df

    # ------------------------------------------------------------------
    def run(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """Run full pipeline and optionally save output."""
        print(f"\u25C9 Running comparison in '{self.mode.upper()}' mode...")
        self.load_data()

        if self.mode in ("tfidf", "both", "all", "llm"):
            # tfidf used also to prefilter for LLM
            print("Computing TF-IDF similarity...")
            self.compute_tfidf_similarity()

        if self.mode in ("semantic", "both", "all", "llm") and self.llm_prefilter == "semantic":
            # If LLM prefilter is semantic or mode explicitly requests semantic
            print("Computing Semantic similarity...")
            self.compute_semantic_similarity()
        elif self.mode in ("semantic", "both", "all"):
            print("Computing Semantic similarity...")
            self.compute_semantic_similarity()

        if self.mode in ("llm", "all"):
            print("Computing LLM-based similarity (reranking top-k)...")
            self.compute_llm_similarity()

        print("Generating results...")
        self.generate_results()

        if output_path:
            self.save_results(output_path)
            print(f"\u2705 Results saved to: {output_path}")
        return self.get_results()
