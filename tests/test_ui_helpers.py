from components.settings import ConfigurationError
from components.ui_helpers import (
    SUGGESTED_QUESTIONS,
    citation_cards,
    response_metrics,
    safe_error_message,
)


def _response():
    return {
        "sources": [
            {
                "title": "CUDA C++ Guide",
                "section": "Streams",
                "source_url": "https://docs.nvidia.com/cuda/",
                "retrieval_score": 0.87654,
                "excerpt": "Streams order work.",
            }
        ],
        "latency": {"retrieval_ms": 2.345, "generation_ms": 4.567},
        "model": "extractive-fallback",
        "corpus_version": "demo-v1",
    }


def test_suggested_questions_cover_three_scripted_demo_topics():
    assert len(SUGGESTED_QUESTIONS) >= 3
    assert any("stream" in question.lower() for question in SUGGESTED_QUESTIONS)
    assert any("occupancy" in question.lower() for question in SUGGESTED_QUESTIONS)
    assert any("pinned" in question.lower() for question in SUGGESTED_QUESTIONS)


def test_citation_cards_and_metrics_are_formatted_without_streamlit():
    cards = citation_cards(_response())
    metrics = response_metrics(_response())

    assert cards == [
        {
            "title": "CUDA C++ Guide",
            "section": "Streams",
            "url": "https://docs.nvidia.com/cuda/",
            "score": "0.877",
            "excerpt": "Streams order work.",
        }
    ]
    assert metrics == {
        "Sources": "1",
        "Retrieval": "2.3 ms",
        "Generation": "4.6 ms",
        "Model": "extractive-fallback",
        "Corpus": "demo-v1",
    }


def test_safe_error_message_allows_validation_detail_but_hides_service_exception():
    assert (
        safe_error_message(ConfigurationError("query cannot exceed 500 characters"))
        == "query cannot exceed 500 characters"
    )
    assert "secret-token" not in safe_error_message(RuntimeError("secret-token leaked"))
    assert "temporarily unavailable" in safe_error_message(RuntimeError("boom"))
