# Text Similarity Comparator

Generic utility to compare two text datasets using TF-IDF, semantic embeddings (sentence-transformers), and optional LLM-based reranking/scoring. It supports flexible column selection and can output enriched results including additional columns from both datasets.

## Features

- **TF-IDF cosine similarity**: Fast lexical similarity.
- **Semantic similarity**: Uses `sentence-transformers` (default: `all-MiniLM-L6-v2`).
- **LLM-based reranking/scoring**: Ask an LLM to score the best candidate among top-k prefiltered matches.
- **Weighted combination**: Combine multiple scores into a single `combined_score`.
- **CLI and Python API**: Use from the command line or as a library.

## Installation

Requires Python 3.9+.

```bash
pip install -e .[dev]
# If you plan to use LLM-based mode via OpenAI
pip install -e .[llm]
```

## Quickstart (CLI)

Prepare two CSV files:
- Source CSV (e.g., `agent_assist_tips.csv`) with a text column like `response`.
- Target CSV (e.g., `guru_cards.csv`) with a text column like `text`.

Run TF-IDF + Semantic (combined):

```bash
text-similarity \
  --from-csv examples/data/agent_assist_tips.csv \
  --to-csv examples/data/guru_cards.csv \
  --from-text response \
  --to-text text \
  --from-cols questions \
  --to-cols title \
  --mode both \
  --output comparison_results.csv
```

Run TF-IDF only:

```bash
text-similarity \
  --from-csv examples/data/agent_assist_tips.csv \
  --to-csv examples/data/guru_cards.csv \
  --from-text response \
  --to-text text \
  --mode tfidf \
  --output tfidf_only_results.csv
```

Run with LLM reranking (OpenAI):

```bash
export OPENAI_API_KEY=...  # required for openai provider
text-similarity \
  --from-csv examples/data/agent_assist_tips.csv \
  --to-csv examples/data/guru_cards.csv \
  --from-text response \
  --to-text text \
  --mode llm \
  --llm-provider openai \
  --openai-model gpt-4o-mini \
  --llm-top-k 3 \
  --output llm_reranked_results.csv
```

You can also use a built-in rule-based scorer (no API key) for offline testing:

```bash
text-similarity \
  --from-csv examples/data/agent_assist_tips.csv \
  --to-csv examples/data/guru_cards.csv \
  --from-text response \
  --to-text text \
  --mode llm \
  --llm-provider rule \
  --llm-top-k 3 \
  --output llm_reranked_results.csv
```

## Python API

```python
from text_similarity import TextSimilarityComparator
from text_similarity.llm import RuleBasedLLMClient

comp = TextSimilarityComparator(
    from_csv_path="examples/data/agent_assist_tips.csv",
    to_csv_path="examples/data/guru_cards.csv",
    from_text_column="response",
    to_text_column="text",
    from_additional_columns=["questions"],
    to_additional_columns=["title"],
    mode="both",  # "tfidf", "semantic", "llm", "both", or "all"
)

df = comp.run(output_path="comparison_results.csv")
print(df.head())

# LLM mode
llm_comp = TextSimilarityComparator(
    from_csv_path="examples/data/agent_assist_tips.csv",
    to_csv_path="examples/data/guru_cards.csv",
    from_text_column="response",
    to_text_column="text",
    mode="llm",
    llm_client=RuleBasedLLMClient(),
    llm_top_k=3,
)
llm_df = llm_comp.run()
```

## Output Columns

- `from_<from_text_column>`: the source text.
- For each score type present:
  - `to_best_match_tfidf`, `tfidf_score`
  - `to_best_match_semantic`, `semantic_score`
  - `to_best_match_llm`, `llm_score`
- Additional columns (if requested): prefixed with `to_` or `from_` and suffixed by score name.
- `combined_score` (if multiple scores available) and `comment` (Aligned/Partial/Misaligned).

## Design Notes

- Semantic mode lazily imports `sentence-transformers` to keep startup fast and optional.
- LLM mode supports an injectable client; `OpenAIJSONSimilarityClient` uses JSON responses for robustness; `RuleBasedLLMClient` is a deterministic stub.
- Weighted combination is a per-row weighted average of available scores.

## Development

Run tests:

```bash
pytest
```

Linting is not configured; feel free to add `ruff`/`black`.

## License

MIT
