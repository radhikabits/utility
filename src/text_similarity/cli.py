from __future__ import annotations

import argparse
from typing import List, Optional

from .comparator import TextSimilarityComparator
from .llm import OpenAIJSONSimilarityClient, RuleBasedLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two CSV datasets using TF-IDF, semantic, and/or LLM reranking."
    )
    parser.add_argument("--from-csv", required=True, help="Path to source CSV")
    parser.add_argument("--to-csv", required=True, help="Path to target CSV")
    parser.add_argument("--from-text", required=True, help="Source text column name")
    parser.add_argument("--to-text", required=True, help="Target text column name")
    parser.add_argument(
        "--from-cols",
        default="",
        help="Comma-separated additional source columns to include in results",
    )
    parser.add_argument(
        "--to-cols",
        default="",
        help="Comma-separated additional target columns to include in results",
    )
    parser.add_argument(
        "--mode",
        choices=["tfidf", "semantic", "llm", "both", "all"],
        default="both",
        help="Which similarity metrics to compute",
    )
    parser.add_argument("--output", default="comparison_results.csv", help="Output CSV path")

    # Weighting
    parser.add_argument("--weight-tfidf", type=float, default=1.0, help="Weight for TF-IDF score")
    parser.add_argument(
        "--weight-semantic", type=float, default=1.0, help="Weight for semantic score"
    )
    parser.add_argument("--weight-llm", type=float, default=1.0, help="Weight for LLM score")

    # LLM config
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "rule"],
        default="rule",
        help="LLM provider (openai requires API key)",
    )
    parser.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--llm-top-k", type=int, default=5, help="Number of candidates to rerank")
    parser.add_argument(
        "--llm-prefilter",
        choices=["tfidf", "semantic"],
        default="tfidf",
        help="Which metric to use to choose candidates before LLM reranking",
    )

    return parser.parse_args()


def build_llm_client(provider: str, model: str):
    if provider == "openai":
        return OpenAIJSONSimilarityClient(model=model)
    return RuleBasedLLMClient()


def main() -> None:
    args = parse_args()
    from_cols: List[str] = [c for c in (args.from_cols.split(",") if args.from_cols else []) if c]
    to_cols: List[str] = [c for c in (args.to_cols.split(",") if args.to_cols else []) if c]

    comparator = TextSimilarityComparator(
        from_csv_path=args.from_csv,
        to_csv_path=args.to_csv,
        from_text_column=args.from_text,
        to_text_column=args.to_text,
        from_additional_columns=from_cols,
        to_additional_columns=to_cols,
        mode=args.mode,
        weight_tfidf=args.weight_tfidf,
        weight_semantic=args.weight_semantic,
        weight_llm=args.weight_llm,
        llm_client=build_llm_client(args.llm_provider, args.openai_model)
        if args.mode in ("llm", "all")
        else None,
        llm_top_k=args.llm_top_k,
        llm_prefilter=args.llm_prefilter,
    )

    comparator.run(output_path=args.output)


if __name__ == "__main__":
    main()
