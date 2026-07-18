"""Deterministic, dependency-free retrieval over the public demo fixture corpus."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


MAX_FIXTURE_BYTES = 1_000_000
MIN_RELEVANCE_SCORE = 0.08
_REQUIRED_FIELDS = {
    "id",
    "chunk_id",
    "title",
    "section",
    "source_url",
    "source_date",
    "license",
    "provenance",
    "content",
}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "cuda",
    "do",
    "does",
    "for",
    "gpu",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "why",
    "with",
}


class FixtureError(ValueError):
    """Raised when a demo fixture cannot satisfy the public corpus contract."""


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOP_WORDS and len(token) > 1
    ]


def load_fixture(path: str | Path) -> tuple[str, tuple[dict[str, Any], ...]]:
    path = Path(path)
    if path.stat().st_size > MAX_FIXTURE_BYTES:
        raise FixtureError("demo fixture must remain under 1 MB")
    with path.open(encoding="utf-8") as fixture:
        payload = json.load(fixture)

    corpus_version = str(payload.get("corpus_version", "")).strip()
    records = payload.get("records")
    if not corpus_version or not isinstance(records, list):
        raise FixtureError("fixture requires corpus_version and records")
    if not 10 <= len(records) <= 20:
        raise FixtureError("demo fixture must contain between 10 and 20 records")

    seen_ids = set()
    seen_chunks = set()
    normalized = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise FixtureError(f"record {index} must be an object")
        missing = sorted(_REQUIRED_FIELDS - record.keys())
        if missing:
            raise FixtureError(f"record {index} is missing: {', '.join(missing)}")
        if not str(record["source_url"]).startswith("https://"):
            raise FixtureError(f"record {index} source_url must use HTTPS")
        if record["id"] in seen_ids or record["chunk_id"] in seen_chunks:
            raise FixtureError("fixture record IDs and chunk IDs must be unique")
        if not _tokens(str(record["content"])):
            raise FixtureError(f"record {index} content cannot be empty")
        seen_ids.add(record["id"])
        seen_chunks.add(record["chunk_id"])
        normalized.append(dict(record, corpus_version=corpus_version))
    return corpus_version, tuple(normalized)


class DemoRetriever:
    """Small TF-IDF cosine retriever with stable score and ID tie-breaking."""

    def __init__(self, fixture_path: str | Path):
        self.corpus_version, self.records = load_fixture(fixture_path)
        self._document_terms = [
            Counter(
                _tokens(
                    f"{record['title']} {record['section']} {record['section']} "
                    f"{record['content']}"
                )
            )
            for record in self.records
        ]
        document_frequency = Counter()
        for terms in self._document_terms:
            document_frequency.update(terms.keys())
        count = len(self.records)
        self._idf = {
            term: math.log((1 + count) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }

    def _vector(self, terms: Counter[str]) -> dict[str, float]:
        return {
            term: frequency * self._idf.get(term, 0.0)
            for term, frequency in terms.items()
            if term in self._idf
        }

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(value * right.get(term, 0.0) for term, value in left.items())
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / (left_norm * right_norm) if numerator else 0.0

    def retrieve_and_rerank(self, query: str, top_k: int = 3) -> dict[str, Any]:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        query_vector = self._vector(Counter(_tokens(query)))
        scored = []
        for record, terms in zip(self.records, self._document_terms):
            score = self._cosine(query_vector, self._vector(terms))
            if score >= MIN_RELEVANCE_SCORE:
                scored.append((score, str(record["chunk_id"]), record))
        scored.sort(key=lambda item: (-item[0], item[1]))

        results = [
            {
                "document_id": str(record["id"]),
                "chunk_id": str(record["chunk_id"]),
                "document": str(record["content"]),
                "source_url": str(record["source_url"]),
                "title": str(record["title"]),
                "section": str(record["section"]),
                "source_date": str(record["source_date"]),
                "license": str(record["license"]),
                "provenance": str(record["provenance"]),
                "corpus_version": self.corpus_version,
                "score": round(score, 6),
            }
            for score, _, record in scored[:top_k]
        ]
        return {
            "original_query": query,
            "results": results,
            "candidate_count": len(scored),
            "retrieval_strategy": "deterministic-tfidf-cosine",
            "corpus_version": self.corpus_version,
        }
