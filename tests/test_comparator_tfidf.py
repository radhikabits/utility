from __future__ import annotations

import pandas as pd
from text_similarity.comparator import TextSimilarityComparator


def test_tfidf_best_match(tmp_path):
    from_csv = tmp_path / "from.csv"
    to_csv = tmp_path / "to.csv"
    from_csv.write_text(
        "response,questions\n"
        "Reset your password using the email link.,What if I forget my password?\n"
        "Change billing info in settings.,How to update billing?\n"
    )
    to_csv.write_text(
        "title,text\n"
        "Password Reset,Reset your password by email link.\n"
        "Billing,Update billing details from your account page.\n"
    )

    comp = TextSimilarityComparator(
        from_csv_path=str(from_csv),
        to_csv_path=str(to_csv),
        from_text_column="response",
        to_text_column="text",
        from_additional_columns=["questions"],
        to_additional_columns=["title"],
        mode="tfidf",
    )
    df = comp.run()

    # Expect the two rows to match in order
    assert list(df["to_best_match_tfidf"]) == [
        "Reset your password by email link.",
        "Update billing details from your account page.",
    ]
    assert "tfidf_score" in df.columns
    assert df.shape[0] == 2
