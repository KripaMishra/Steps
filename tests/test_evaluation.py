from components.evaluate import evaluate_questions, load_questions


def _response(*, source=True, insufficient=False, score=0.9):
    sources = []
    if source:
        sources = [
            {
                "source_url": "https://example.com/cuda",
                "title": "CUDA guide",
                "section": "Streams",
                "chunk_id": "streams",
                "document_id": "1",
                "retrieval_score": score,
                "excerpt": "text",
                "source_date": "2026-07-18",
                "provenance": "summary",
            }
        ]
    return {
        "query": "q",
        "answer": "Insufficient context: no match" if insufficient else "answer",
        "sources": sources,
        "mode": "demo",
        "latency": {"retrieval_ms": 1.0, "generation_ms": 2.0, "total_ms": 3.0},
        "retrieval_metadata": {},
        "model": "extractive-fallback",
        "corpus_version": "demo-v1",
        "insufficient_context": insufficient,
    }


def test_evaluator_reports_hits_citations_insufficient_context_and_schema():
    questions = [
        {
            "id": "answerable",
            "question": "streams",
            "expected_source_ids": ["streams"],
            "expected_source_urls": [],
            "expect_insufficient_context": False,
        },
        {
            "id": "unknown",
            "question": "unknown",
            "expected_source_ids": [],
            "expected_source_urls": [],
            "expect_insufficient_context": True,
        },
    ]

    report = evaluate_questions(
        questions,
        lambda question: _response(source=question == "streams", insufficient=question == "unknown"),
    )

    assert report["retrieval_hit_rate"] == 1.0
    assert report["citation_presence_rate"] == 1.0
    assert report["insufficient_context_accuracy"] == 1.0
    assert report["schema_valid_rate"] == 1.0
    assert report["latency_valid_rate"] == 1.0


def test_golden_question_file_covers_answerable_ambiguous_and_unanswerable():
    questions = load_questions("eval/questions.json")

    assert any(question["expected_source_ids"] for question in questions)
    assert sum(question["expect_insufficient_context"] for question in questions) >= 2
