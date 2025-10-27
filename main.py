"""
Example usage for the TextSimilarityComparator utility.

Compares two CSV files (e.g., Guru Cards and Agent Assist Tips)
using TF-IDF, Semantic similarity, and optional LLM reranking.
"""
from __future__ import annotations

from text_similarity.comparator import TextSimilarityComparator
from text_similarity.llm import RuleBasedLLMClient


def run_examples() -> None:
    print("\n\U0001F680 Example 1: Compare Agent Assist Tips vs Guru Cards (TF-IDF & Semantic)\n")
    comparator = TextSimilarityComparator(
        from_csv_path="examples/data/agent_assist_tips.csv",
        to_csv_path="examples/data/guru_cards.csv",
        from_text_column="response",
        to_text_column="text",
        from_additional_columns=["questions"],
        to_additional_columns=["title"],
        mode="both",
    )
    comparator.run(output_path="comparison_results.csv")
    print("\n\u2705 Combined comparison complete! Results saved to 'comparison_results.csv'.")

    print("\n\U0001F680 Example 2: TF-IDF only comparison (lightweight mode)\n")
    tfidf_comparator = TextSimilarityComparator(
        from_csv_path="examples/data/agent_assist_tips.csv",
        to_csv_path="examples/data/guru_cards.csv",
        from_text_column="response",
        to_text_column="text",
        from_additional_columns=["questions"],
        to_additional_columns=["title"],
        mode="tfidf",
    )
    tfidf_comparator.run(output_path="tfidf_only_results.csv")
    print("\u2705 TF-IDF only comparison complete! Results saved to 'tfidf_only_results.csv'.")

    print("\n\U0001F680 Example 3: LLM reranking over TF-IDF top-k\n")
    llm_comparator = TextSimilarityComparator(
        from_csv_path="examples/data/agent_assist_tips.csv",
        to_csv_path="examples/data/guru_cards.csv",
        from_text_column="response",
        to_text_column="text",
        from_additional_columns=["questions"],
        to_additional_columns=["title"],
        mode="llm",
        llm_client=RuleBasedLLMClient(),  # Replace with OpenAIJSONSimilarityClient for real LLM
        llm_top_k=3,
    )
    llm_comparator.run(output_path="llm_reranked_results.csv")
    print("\u2705 LLM reranking comparison complete! Results saved to 'llm_reranked_results.csv'.")


if __name__ == "__main__":
    run_examples()
