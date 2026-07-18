"""Runtime configuration and trust-boundary validation for the RAG application."""

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
    rag_mode: str = "demo"
    query_max_chars: int = 500
    top_k_max: int = 10
    generation_max_tokens: int = 512
    request_timeout_seconds: int = 60
    demo_fixture_path: Path = Path("demo/fixtures/cuda_docs.json")


def _int_setting(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name, str(default)).strip()
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be greater than zero")
    return parsed


def _required_text(env: Mapping[str, str], name: str, default: str) -> str:
    value = env.get(name, default).strip()
    if not value:
        raise ConfigurationError(f"{name} cannot be empty")
    return value


def load_settings(
    *, require_gemini: bool | None = None, env: Mapping[str, str] | None = None
) -> Settings:
    """Load settings; Gemini is optional in demo mode and required in full mode."""

    values = os.environ if env is None else env
    rag_mode = values.get("RAG_MODE", "demo").strip()
    if rag_mode not in {"demo", "full"}:
        raise ConfigurationError("RAG_MODE must be 'demo' or 'full'")

    api_key = values.get("GEMINI_API_KEY", "").strip() or None
    should_require_gemini = rag_mode == "full" if require_gemini is None else require_gemini
    if should_require_gemini and api_key is None:
        raise ConfigurationError(
            "GEMINI_API_KEY is required to use the Gemini answer generator"
        )

    return Settings(
        gemini_api_key=api_key,
        gemini_model=_required_text(values, "GEMINI_MODEL", "gemini-2.5-flash"),
        gemini_base_url=_required_text(values, "GEMINI_BASE_URL", GEMINI_BASE_URL),
        milvus_host=_required_text(values, "MILVUS_HOST", "localhost"),
        milvus_port=_int_setting(values, "MILVUS_PORT", 19530),
        elasticsearch_host=_required_text(values, "ELASTICSEARCH_HOST", "localhost"),
        elasticsearch_port=_int_setting(values, "ELASTICSEARCH_PORT", 9200),
        collection_name=_required_text(values, "MILVUS_COLLECTION", "Test_collection"),
        elasticsearch_index=_required_text(values, "ELASTICSEARCH_INDEX", "documents"),
        result_dir=Path(_required_text(values, "RAG_RESULT_DIR", "result")),
        context_max_chars=_int_setting(values, "RAG_CONTEXT_MAX_CHARS", 12000),
        rag_mode=rag_mode,
        query_max_chars=_int_setting(values, "RAG_QUERY_MAX_CHARS", 500),
        top_k_max=_int_setting(values, "RAG_TOP_K_MAX", 10),
        generation_max_tokens=_int_setting(values, "RAG_GENERATION_MAX_TOKENS", 512),
        request_timeout_seconds=_int_setting(values, "RAG_REQUEST_TIMEOUT_SECONDS", 60),
        demo_fixture_path=Path(
            _required_text(values, "RAG_DEMO_FIXTURE_PATH", "demo/fixtures/cuda_docs.json")
        ),
    )


def validate_query(query: str, settings: Settings) -> str:
    """Normalize and validate a user query at the application boundary."""

    if not isinstance(query, str):
        raise ConfigurationError("query must be text")
    normalized = query.strip()
    if not normalized:
        raise ConfigurationError("query cannot be empty")
    if len(normalized) > settings.query_max_chars:
        raise ConfigurationError(
            f"query cannot exceed {settings.query_max_chars} characters"
        )
    return normalized


def validate_top_k(top_k: int, settings: Settings) -> int:
    """Validate a retrieval count before it reaches a backend."""

    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ConfigurationError("top_k must be an integer")
    if not 1 <= top_k <= settings.top_k_max:
        raise ConfigurationError(f"top_k must be between 1 and {settings.top_k_max}")
    return top_k
