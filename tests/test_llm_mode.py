from __future__ import annotations

from text_similarity.comparator import TextSimilarityComparator
from text_similarity.llm import RuleBasedLLMClient


def test_llm_reranking_top1(tmp_path):
    from_csv = tmp_path / "from.csv"
    to_csv = tmp_path / "to.csv"
    from_csv.write_text(
        "response\n"
        "Reset your password using the email link.\n"
        "Change billing info in settings.\n"
    )
    to_csv.write_text(
        "text\n"
        "Reset your password by email link.\n"
        "Update billing details from your account page.\n"
    )

    comp = TextSimilarityComparator(
        from_csv_path=str(from_csv),
        to_csv_path=str(to_csv),
        from_text_column="response",
        to_text_column="text",
        mode="llm",
        llm_client=RuleBasedLLMClient(),
        llm_top_k=1,  # use only the top TF-IDF candidate
    )
    df = comp.run()

    # With top_k=1, LLM picks the same as TF-IDF best
    assert list(df["to_best_match_llm"]) == [
        "Reset your password by email link.",
        "Update billing details from your account page.",
    ]
    assert "llm_score" in df.columns
