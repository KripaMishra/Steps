"""Serializable response types shared by demo and full retrieval modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


INSUFFICIENT_CONTEXT_ANSWER = (
    "Insufficient context: the indexed CUDA sources do not support an answer."
)


@dataclass(frozen=True)
class Source:
    source_url: str
    title: str
    section: str
    chunk_id: str
    document_id: str
    retrieval_score: float
    excerpt: str = ""
    source_date: str = ""
    provenance: str = ""

    def __post_init__(self) -> None:
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("source_url must be an HTTP(S) URL")
        if not self.title.strip() or not self.chunk_id.strip():
            raise ValueError("source title and chunk_id are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_url": self.source_url,
            "title": self.title,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "retrieval_score": float(self.retrieval_score),
            "excerpt": self.excerpt,
            "source_date": self.source_date,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Source":
        return cls(
            source_url=str(value["source_url"]),
            title=str(value["title"]),
            section=str(value.get("section", "")),
            chunk_id=str(value["chunk_id"]),
            document_id=str(value.get("document_id", value["chunk_id"])),
            retrieval_score=float(value["retrieval_score"]),
            excerpt=str(value.get("excerpt", "")),
            source_date=str(value.get("source_date", "")),
            provenance=str(value.get("provenance", "")),
        )


@dataclass(frozen=True)
class Latency:
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

    def __post_init__(self) -> None:
        if min(self.retrieval_ms, self.generation_ms, self.total_ms) < 0:
            raise ValueError("latency values cannot be negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_ms": round(float(self.retrieval_ms), 3),
            "generation_ms": round(float(self.generation_ms), 3),
            "total_ms": round(float(self.total_ms), 3),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Latency":
        return cls(
            retrieval_ms=float(value.get("retrieval_ms", 0.0)),
            generation_ms=float(value.get("generation_ms", 0.0)),
            total_ms=float(value.get("total_ms", 0.0)),
        )


@dataclass(frozen=True)
class RAGResponse:
    query: str
    answer: str
    sources: tuple[Source, ...]
    mode: str
    latency: Latency
    retrieval_metadata: Mapping[str, Any] = field(default_factory=dict)
    model: str = ""
    corpus_version: str = ""
    insufficient_context: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"demo", "full"}:
            raise ValueError("mode must be 'demo' or 'full'")
        if not self.query.strip() or not self.answer.strip():
            raise ValueError("query and answer are required")
        if not self.sources and not self.insufficient_context:
            raise ValueError("responses without sources must set insufficient_context")

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [source.to_dict() for source in self.sources],
            "mode": self.mode,
            "latency": self.latency.to_dict(),
            "retrieval_metadata": dict(self.retrieval_metadata),
            "model": self.model,
            "corpus_version": self.corpus_version,
            "insufficient_context": self.insufficient_context,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RAGResponse":
        return cls(
            query=str(value["query"]),
            answer=str(value["answer"]),
            sources=tuple(Source.from_dict(source) for source in value.get("sources", [])),
            mode=str(value["mode"]),
            latency=Latency.from_dict(value.get("latency", {})),
            retrieval_metadata=dict(value.get("retrieval_metadata", {})),
            model=str(value.get("model", "")),
            corpus_version=str(value.get("corpus_version", "")),
            insufficient_context=bool(value.get("insufficient_context", False)),
        )

    @classmethod
    def insufficient(
        cls,
        *,
        query: str,
        mode: str,
        retrieval_ms: float = 0.0,
        total_ms: float = 0.0,
        corpus_version: str = "",
        model: str = "extractive-fallback",
        retrieval_metadata: Mapping[str, Any] | None = None,
    ) -> "RAGResponse":
        return cls(
            query=query,
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            sources=(),
            mode=mode,
            latency=Latency(retrieval_ms=retrieval_ms, total_ms=total_ms),
            retrieval_metadata=retrieval_metadata or {},
            model=model,
            corpus_version=corpus_version,
            insufficient_context=True,
        )
