import json

import pytest

from components.response_schema import Latency, RAGResponse, Source


def test_response_round_trip_preserves_sources_scores_mode_and_latency():
    response = RAGResponse(
        query="What is a CUDA stream?",
        answer="A stream orders GPU work.",
        sources=(
            Source(
                source_url="https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
                title="CUDA C++ Programming Guide",
                section="Streams",
                chunk_id="programming-guide-streams",
                document_id="1",
                retrieval_score=0.91,
                excerpt="A stream is a sequence of commands.",
            ),
        ),
        mode="demo",
        latency=Latency(retrieval_ms=2.5, generation_ms=0.2, total_ms=2.7),
        retrieval_metadata={"top_k": 3},
        model="extractive-fallback",
        corpus_version="2026-07-18",
    )

    payload = response.to_dict()
    restored = RAGResponse.from_dict(json.loads(json.dumps(payload)))

    assert restored == response
    assert payload["sources"][0]["retrieval_score"] == 0.91
    assert payload["mode"] == "demo"
    assert payload["latency"]["total_ms"] == 2.7


def test_empty_sources_require_insufficient_context():
    with pytest.raises(ValueError, match="insufficient_context"):
        RAGResponse(
            query="unknown",
            answer="made up",
            sources=(),
            mode="demo",
            latency=Latency(),
            insufficient_context=False,
        )


def test_insufficient_context_response_serializes_without_sources():
    response = RAGResponse.insufficient(
        query="Who won the World Cup?",
        mode="demo",
        retrieval_ms=1.0,
        total_ms=1.2,
        corpus_version="2026-07-18",
    )

    assert response.sources == ()
    assert response.insufficient_context is True
    assert response.to_dict()["answer"].startswith("Insufficient context")
