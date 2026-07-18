"""Pure presentation helpers for the Streamlit demo."""

from __future__ import annotations

from typing import Any

from components.settings import ConfigurationError


SUGGESTED_QUESTIONS = (
    "How do CUDA streams allow work to overlap?",
    "What factors limit CUDA occupancy?",
    "Why is pinned host memory used for asynchronous copies?",
)


def citation_cards(response: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "title": str(source.get("title", "Untitled source")),
            "section": str(source.get("section", "")),
            "url": str(source.get("source_url", "")),
            "score": f"{float(source.get('retrieval_score', 0.0)):.3f}",
            "excerpt": str(source.get("excerpt", "")),
        }
        for source in response.get("sources", [])
    ]


def response_metrics(response: dict[str, Any]) -> dict[str, str]:
    latency = response.get("latency", {})
    return {
        "Sources": str(len(response.get("sources", []))),
        "Retrieval": f"{float(latency.get('retrieval_ms', 0.0)):.1f} ms",
        "Generation": f"{float(latency.get('generation_ms', 0.0)):.1f} ms",
        "Model": str(response.get("model", "unknown")),
        "Corpus": str(response.get("corpus_version", "unknown") or "unknown"),
    }


def safe_error_message(error: Exception) -> str:
    if isinstance(error, ConfigurationError):
        return str(error)
    return "The answer service is temporarily unavailable. Please try again."
