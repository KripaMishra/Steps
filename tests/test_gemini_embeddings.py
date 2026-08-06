import sys
import types

from components.gemini_embeddings import GeminiEmbedder
from components.settings import load_settings


def test_gemini_embedder_uses_retrieval_tasks_and_configured_dimensions(monkeypatch):
    calls = []

    class EmbedContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeClient:
        def __init__(self, *, api_key):
            assert api_key == "test-key"
            self.models = self

        def embed_content(self, **kwargs):
            calls.append(kwargs)
            return types.SimpleNamespace(
                embeddings=[types.SimpleNamespace(values=[0.1, 0.2, 0.3])]
            )

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = FakeClient
    genai.types = types.SimpleNamespace(EmbedContentConfig=EmbedContentConfig)
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    settings = load_settings(
        env={
            "GEMINI_API_KEY": "test-key",
            "GEMINI_EMBEDDING_DIMENSIONS": "768",
        }
    )
    embedder = GeminiEmbedder(settings)

    assert embedder.embed_document("CUDA stream") == [0.1, 0.2, 0.3]
    assert embedder.embed_query("What is a CUDA stream?") == [0.1, 0.2, 0.3]
    assert [call["config"].kwargs["task_type"] for call in calls] == [
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
    ]
    assert all(call["config"].kwargs["output_dimensionality"] == 768 for call in calls)
