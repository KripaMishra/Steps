"""Runtime configuration for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class ConfigurationError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    gemini_base_url: str
    milvus_host: str
    milvus_port: int
    elasticsearch_host: str
    elasticsearch_port: int
    collection_name: str
    elasticsearch_index: str
    result_dir: Path
    context_max_chars: int


def _int_setting(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return parsed


def load_settings(
    *, require_gemini: bool = True, env: Mapping[str, str] | None = None
) -> Settings:
    """Load application settings from environment variables."""

    values = os.environ if env is None else env
    api_key = values.get("GEMINI_API_KEY", "").strip() or None
    if require_gemini and api_key is None:
        raise ConfigurationError(
            "GEMINI_API_KEY is required to use the Gemini answer generator"
        )

    return Settings(
        gemini_api_key=api_key,
        gemini_model=values.get("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        gemini_base_url=values.get("GEMINI_BASE_URL", GEMINI_BASE_URL).strip(),
        milvus_host=values.get("MILVUS_HOST", "localhost").strip(),
        milvus_port=_int_setting(values, "MILVUS_PORT", 19530),
        elasticsearch_host=values.get("ELASTICSEARCH_HOST", "localhost").strip(),
        elasticsearch_port=_int_setting(values, "ELASTICSEARCH_PORT", 9200),
        collection_name=values.get("MILVUS_COLLECTION", "Test_collection").strip(),
        elasticsearch_index=values.get("ELASTICSEARCH_INDEX", "documents").strip(),
        result_dir=Path(values.get("RAG_RESULT_DIR", "result").strip()),
        context_max_chars=_int_setting(values, "RAG_CONTEXT_MAX_CHARS", 12000),
    )
