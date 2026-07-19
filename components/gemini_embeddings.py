"""Gemini embeddings for Milvus document and query vectors."""

from __future__ import annotations

from typing import Literal

from components.settings import ConfigurationError, Settings


TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


class GeminiEmbedder:
    """Generate compact Gemini vectors without local model downloads."""

    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise ConfigurationError("GEMINI_API_KEY is required for embeddings")

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model
        self._dimensions = settings.gemini_embedding_dimensions

    def embed(self, text: str, *, task_type: TaskType) -> list[float]:
        response = self._client.models.embed_content(
            model=self._model,
            contents=text,
            config=self._types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dimensions,
            ),
        )
        if not response.embeddings or not response.embeddings[0].values:
            raise RuntimeError("Gemini returned an empty embedding")
        return list(response.embeddings[0].values)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text, task_type="RETRIEVAL_QUERY")
