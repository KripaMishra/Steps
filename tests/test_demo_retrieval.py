import json
from pathlib import Path

import pytest

from components.demo_retrieval import DemoRetriever, FixtureError, load_fixture


FIXTURE = Path("demo/fixtures/cuda_docs.json")


def test_fixture_is_small_and_has_complete_provenance():
    corpus_version, records = load_fixture(FIXTURE)

    assert FIXTURE.stat().st_size < 1_000_000
    assert corpus_version == "2026-07-18.demo-v1"
    assert 10 <= len(records) <= 20
    assert all(record["source_url"].startswith("https://docs.nvidia.com/") for record in records)
    assert all(record["source_date"] and record["license"] and record["provenance"] for record in records)


def test_demo_retrieval_returns_relevant_cited_records_deterministically():
    retriever = DemoRetriever(FIXTURE)

    first = retriever.retrieve_and_rerank("How can CUDA streams overlap GPU work?", top_k=3)
    second = retriever.retrieve_and_rerank("How can CUDA streams overlap GPU work?", top_k=3)

    assert first == second
    assert first["results"][0]["chunk_id"] == "asynchronous-streams"
    assert first["results"][0]["source_url"].startswith("https://")
    assert first["results"][0]["score"] > 0
    assert first["corpus_version"] == "2026-07-18.demo-v1"


def test_demo_retrieval_returns_no_context_for_unrelated_question():
    result = DemoRetriever(FIXTURE).retrieve_and_rerank(
        "Who won the international football tournament?", top_k=3
    )

    assert result["results"] == []
    assert result["candidate_count"] == 0


def test_invalid_fixture_fails_clearly(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"corpus_version": "x", "records": [{"id": i} for i in range(10)]}))

    with pytest.raises(FixtureError, match="missing"):
        load_fixture(path)
