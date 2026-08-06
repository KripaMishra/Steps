import importlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest


def _settings(mode="full", api_key="test-key"):
    from components.settings import Settings

    return Settings(
        gemini_api_key=api_key,
        gemini_model="gemini-test",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        milvus_endpoint="https://cloud.example",
        milvus_token="token-value",
        elasticsearch_host="elasticsearch",
        elasticsearch_port=9200,
        collection_name="collection",
        elasticsearch_index="documents",
        result_dir=Path("result"),
        context_max_chars=500,
        rag_mode=mode,
        request_timeout_seconds=12,
        generation_max_tokens=128,
    )


def _retrieval_result(results=None):
    if results is None:
        results = [
            {
                "document": "CUDA uses kernels.",
                "score": 0.9,
                "source_url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
                "title": "CUDA C++ Programming Guide",
                "section": "Kernels",
                "chunk_id": "kernels",
                "document_id": "1",
                "source_date": "2026-07-18",
                "provenance": "Project-authored summary.",
                "corpus_version": "demo-v1",
            }
        ]
    return {
        "results": results,
        "corpus_version": "demo-v1",
        "candidate_count": len(results),
        "retrieval_strategy": "fake",
    }


class FakeRetriever:
    def __init__(self, result=None):
        self.result = result if result is not None else _retrieval_result()
        self.retrieve_calls = 0
        self.setup_kwargs = None

    def setup(self, **kwargs):
        self.setup_kwargs = kwargs

    def retrieve_and_rerank(self, query, top_k=3):
        self.retrieve_calls += 1
        return self.result


class FakeLLM:
    def __init__(self, content="Grounded answer"):
        self.content = content
        self.invoke_calls = []

    def invoke(self, prompt):
        self.invoke_calls.append(prompt)
        return types.SimpleNamespace(content=self.content)


def _load_module():
    sys.modules.pop("components.s5_rag_llm", None)
    return importlib.import_module("components.s5_rag_llm")


def test_full_mode_retrieves_once_and_returns_shared_response_contract():
    module = _load_module()
    retriever = FakeRetriever()
    llm = FakeLLM()
    model = module.RAGModel(
        "legacy-model", settings=_settings(), retriever=retriever, llm=llm
    )

    result = model.process_query("What are CUDA kernels?")

    assert retriever.retrieve_calls == 1
    assert result["answer"] == "Grounded answer"
    assert result["mode"] == "full"
    assert result["sources"][0]["chunk_id"] == "kernels"
    assert result["model"] == "gemini-test"
    assert result["latency"]["total_ms"] >= 0
    assert len(llm.invoke_calls) == 1
    assert retriever.setup_kwargs == {
        "milvus_endpoint": "https://cloud.example",
        "milvus_token": "token-value",
        "es_host": "elasticsearch",
        "es_port": 9200,
        "collection_name": "collection",
        "es_index": "documents",
    }


def test_demo_offline_fallback_is_labeled_and_cited():
    module = _load_module()
    retriever = FakeRetriever()
    model = module.RAGModel(
        settings=_settings(mode="demo", api_key=None), retriever=retriever
    )

    result = model.process_query("What are CUDA kernels?")

    assert result["answer"].startswith("Offline extractive fallback:")
    assert result["answer"].endswith("CUDA uses kernels.")
    assert result["model"] == "extractive-fallback"
    assert result["sources"][0]["source_url"].startswith("https://")
    assert result["insufficient_context"] is False
    assert retriever.setup_kwargs is None


def test_empty_retrieval_returns_deterministic_insufficient_context_without_llm():
    module = _load_module()
    retriever = FakeRetriever(_retrieval_result([]))
    llm = FakeLLM()
    model = module.RAGModel(
        settings=_settings(mode="demo"), retriever=retriever, llm=llm
    )

    result = model.process_query("Unknown question")

    assert result["insufficient_context"] is True
    assert result["sources"] == []
    assert result["answer"].startswith("Insufficient context")
    assert llm.invoke_calls == []


def test_context_formatter_supports_multiple_documents_and_limits_size():
    module = _load_module()
    model = module.RAGModel(
        settings=_settings(), retriever=FakeRetriever(), llm=FakeLLM()
    )

    context = model._format_context(
        {
            "results": [
                {"document": "A" * 300, "score": 0.9},
                {"document": "B" * 300, "score": 0.8},
            ]
        }
    )

    assert "Document 1" in context
    assert "Document 2" in context
    assert len(context) <= _settings().context_max_chars


def test_query_and_top_k_limits_are_enforced_before_retrieval():
    module = _load_module()
    retriever = FakeRetriever()
    model = module.RAGModel(
        settings=replace(_settings(), query_max_chars=10, top_k_max=2),
        retriever=retriever,
        llm=FakeLLM(),
    )

    with pytest.raises(module.ConfigurationError):
        model.process_query("x" * 11)
    with pytest.raises(module.ConfigurationError):
        model.process_query("valid", top_k=3)
    assert retriever.retrieve_calls == 0


def test_save_query_results_uses_configured_result_directory(tmp_path):
    module = _load_module()
    settings = replace(_settings(), result_dir=tmp_path)
    model = module.RAGModel(settings=settings, retriever=FakeRetriever(), llm=FakeLLM())

    saved = model.save_query_results({"answer": "ok"})

    assert saved.parent == tmp_path
    assert json.loads(saved.read_text()) == {"answer": "ok"}


def test_gemini_client_receives_explicit_limits(monkeypatch):
    module = _load_module()
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_langchain = types.ModuleType("langchain_openai")
    fake_langchain.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", fake_langchain)

    model = module.RAGModel(settings=_settings(), retriever=FakeRetriever())

    assert model.llm.__class__ is FakeChatOpenAI
    assert captured == {
        "model": "gemini-test",
        "api_key": "test-key",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "temperature": 0.1,
        "max_tokens": 128,
        "timeout": 12,
    }
