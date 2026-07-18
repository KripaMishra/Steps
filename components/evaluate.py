"""Offline regression evaluator for the deterministic demo corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

from components.response_schema import RAGResponse
from components.s5_rag_llm import RAGModel
from components.settings import load_settings


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        questions = json.load(source)
    if not isinstance(questions, list) or not questions:
        raise ValueError("evaluation questions must be a non-empty list")
    return questions


def evaluate_questions(
    questions: Iterable[dict[str, Any]],
    answer: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    details = []
    answerable_hits = 0
    answerable_count = 0
    citation_count = 0
    insufficient_correct = 0
    insufficient_count = 0
    schema_valid = 0
    latency_valid = 0

    for item in questions:
        response = answer(str(item["question"]))
        expected_ids = set(item.get("expected_source_ids", []))
        expected_urls = set(item.get("expected_source_urls", []))
        expect_insufficient = bool(item.get("expect_insufficient_context", False))
        sources = response.get("sources", [])
        source_ids = {source.get("chunk_id") for source in sources}
        source_urls = {source.get("source_url") for source in sources}

        hit = bool(expected_ids & source_ids or expected_urls & source_urls)
        if not expect_insufficient:
            answerable_count += 1
            answerable_hits += int(hit)
            citation_count += int(bool(sources))
        else:
            insufficient_count += 1
            insufficient_correct += int(bool(response.get("insufficient_context")))

        try:
            RAGResponse.from_dict(response)
            valid_schema = True
            schema_valid += 1
        except (KeyError, TypeError, ValueError):
            valid_schema = False

        latency = response.get("latency", {})
        valid_latency = all(
            isinstance(latency.get(field), (int, float)) and latency[field] >= 0
            for field in ("retrieval_ms", "generation_ms", "total_ms")
        ) and latency.get("total_ms", -1) >= latency.get("retrieval_ms", 0)
        latency_valid += int(valid_latency)
        details.append(
            {
                "id": item.get("id", ""),
                "hit": hit,
                "has_citation": bool(sources),
                "insufficient_context": bool(response.get("insufficient_context")),
                "schema_valid": valid_schema,
                "latency_valid": valid_latency,
            }
        )

    total = len(details)
    return {
        "question_count": total,
        "retrieval_hit_rate": answerable_hits / answerable_count if answerable_count else 1.0,
        "citation_presence_rate": citation_count / answerable_count if answerable_count else 1.0,
        "insufficient_context_accuracy": insufficient_correct / insufficient_count if insufficient_count else 1.0,
        "schema_valid_rate": schema_valid / total,
        "latency_valid_rate": latency_valid / total,
        "details": details,
    }


def run_offline(path: str | Path) -> dict[str, Any]:
    settings = replace(
        load_settings(require_gemini=False), rag_mode="demo", gemini_api_key=None
    )
    model = RAGModel(settings=settings)
    return evaluate_questions(
        load_questions(path), lambda question: model.process_query(question, top_k=3)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate offline demo retrieval and citations")
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--min-hit-rate", type=float, default=0.75)
    args = parser.parse_args()
    report = run_offline(args.questions)
    print(json.dumps(report, indent=2))
    passes = (
        report["retrieval_hit_rate"] >= args.min_hit_rate
        and report["citation_presence_rate"] == 1.0
        and report["insufficient_context_accuracy"] == 1.0
        and report["schema_valid_rate"] == 1.0
        and report["latency_valid_rate"] == 1.0
    )
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
