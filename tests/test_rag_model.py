import importlib
import sys
import types
from pathlib import Path


def _settings():
    from components.settings import Settings

    return Settings(
        gemini_api_key="test-key",
        gemini_model="gemini-test",
        gemini_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        milvus_host="milvus",
        milvus_port=19530,
        elasticsearch_host="elasticsearch",
        elasticsearch_port=9200,
        collection_name="collection",
        elasticsearch_index="documents",
        result_dir=Path("result"),
        context_max_chars=500,
    )


class FakeRetriever:
    def __init__(self, result=None):
        self.result = result or {"results": [{"document": "CUDA uses kernels.", "score": 0.9}]}
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


def test_old_positional_model_name_remains_accepted():
    module = _load_module()
    model = module.RAGModel("gpt2-xl", settings=_settings(), retriever=FakeRetriever(), llm=FakeLLM())

    assert model.settings.gemini_model == "gemini-test"


def test_process_query_retrieves_once_and_uses_existing_context():
    module = _load_module()
    retriever = FakeRetriever()
    llm = FakeLLM()
    model = module.RAGModel(settings=_settings(), retriever=retriever, llm=llm)

    result = model.process_query("What are CUDA kernels?")

    assert retriever.retrieve_calls == 1
    assert result["answer"] == "Grounded answer"
    assert len(llm.invoke_calls) == 1
    assert retriever.setup_kwargs == {
        "milvus_host": "milvus",
        "milvus_port": 19530,
        "es_host": "elasticsearch",
        "es_port": 9200,
        "collection_name": "collection",
        "es_index": "documents",
    }


def test_empty_retrieval_does_not_invoke_llm():
    module = _load_module()
    retriever = FakeRetriever({"results": []})
    llm = FakeLLM()
    model = module.RAGModel(settings=_settings(), retriever=retriever, llm=llm)

    answer = model.generate_answer("Unknown question", {"results": []})

    assert "no relevant context" in answer.lower()
    assert llm.invoke_calls == []


def test_context_formatter_supports_multiple_documents_and_limits_size():
    module = _load_module()
    model = module.RAGModel(settings=_settings(), retriever=FakeRetriever(), llm=FakeLLM())

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


def test_save_query_results_uses_configured_result_directory(tmp_path):
    module = _load_module()
    settings = _settings()
    settings = settings.__class__(**{**settings.__dict__, "result_dir": tmp_path})
    model = module.RAGModel(settings=settings, retriever=FakeRetriever(), llm=FakeLLM())

    model.save_query_results({"answer": "ok"})

    saved = list(tmp_path.glob("query_results_*.json"))
    assert len(saved) == 1
    assert '"answer": "ok"' in saved[0].read_text()


def test_gemini_client_receives_explicit_configuration(monkeypatch):
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
    }
