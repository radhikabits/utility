from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Optional
import json


class LLMClient(Protocol):
    def score_similarity(self, source_text: str, target_text: str) -> float:
        """Return a similarity score in [0, 1] between two strings."""
        ...


@dataclass
class RuleBasedLLMClient:
    """A simple local similarity scorer as a drop-in for tests/offline.

    Uses token Jaccard overlap as a crude similarity proxy to avoid external calls.
    """

    lowercase: bool = True

    def score_similarity(self, source_text: str, target_text: str) -> float:
        a = source_text or ""
        b = target_text or ""
        if self.lowercase:
            a = a.lower()
            b = b.lower()
        aset = set(a.split())
        bset = set(b.split())
        if not aset and not bset:
            return 1.0
        if not aset or not bset:
            return 0.0
        inter = len(aset & bset)
        union = len(aset | bset)
        return inter / union


@dataclass
class OpenAIJSONSimilarityClient:
    """LLM client that asks OpenAI Chat Completions for a numeric similarity in [0,1].

    This client requires the `openai` package installed and the environment variable
    OPENAI_API_KEY to be set. It uses JSON response format for robust parsing.
    """

    model: str = "gpt-4o-mini"
    system_prompt: str = (
        "You are a helpful assistant that rates how similar two texts are. "
        "Return strictly a JSON object with a single key 'similarity' where the value is a float between 0 and 1."
    )

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "openai package not available. Install with `pip install text-similarity-comparator[llm]` or `pip install openai`."
            ) from exc
        self._client = OpenAI()

    def score_similarity(self, source_text: str, target_text: str) -> float:
        from openai import OpenAI  # local import for type hints only

        prompt = (
            "Rate the semantic similarity between Text A and Text B as a number between 0 and 1.\n"
            "- 1 means identical or fully aligned.\n"
            "- 0 means completely unrelated.\n"
            "Respond ONLY with a JSON object like {\"similarity\": 0.82}.\n\n"
            f"Text A: {source_text}\n\nText B: {target_text}"
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(content)
            score = float(data.get("similarity", 0.0))
        except Exception:
            score = 0.0
        # Clamp to [0,1]
        return max(0.0, min(1.0, score))
