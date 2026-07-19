from dataclasses import replace

import pytest

from components.settings import (
    ConfigurationError,
    load_settings,
    validate_query,
    validate_top_k,
)


def test_missing_gemini_key_is_allowed_in_demo_mode():
    settings = load_settings(env={})

    assert settings.rag_mode == "demo"
    assert settings.gemini_api_key is None


def test_missing_gemini_key_fails_in_full_mode():
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        load_settings(env={"RAG_MODE": "full"})


def test_settings_use_demo_defaults_and_configured_backend_index():
    settings = load_settings(env={"GEMINI_API_KEY": "test-key"})

    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "gemini-3.1-flash-lite"
    assert settings.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.milvus_endpoint is None
    assert settings.milvus_token is None
    assert settings.elasticsearch_host == "localhost"
    assert settings.elasticsearch_port == 9200
    assert settings.collection_name == "Test_collection"
    assert settings.elasticsearch_index == "documents"
    assert settings.query_max_chars == 500
    assert settings.top_k_max == 10
    assert settings.context_max_chars == 12000
    assert settings.generation_max_tokens == 512


@pytest.mark.parametrize("mode", ["invalid", "DEMO "])
def test_invalid_mode_fails(mode):
    with pytest.raises(ConfigurationError, match="RAG_MODE"):
        load_settings(env={"RAG_MODE": mode})


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_QUERY_MAX_CHARS", "0"),
        ("RAG_TOP_K_MAX", "none"),
        ("RAG_CONTEXT_MAX_CHARS", "-1"),
        ("RAG_GENERATION_MAX_TOKENS", "0"),
        ("RAG_REQUEST_TIMEOUT_SECONDS", "0"),
    ],
)
def test_positive_integer_limits_are_validated(name, value):
    with pytest.raises(ConfigurationError, match=name):
        load_settings(env={name: value})


def test_query_and_top_k_are_validated_at_boundary():
    settings = load_settings(env={})

    assert validate_query("  CUDA streams?  ", settings) == "CUDA streams?"
    assert validate_top_k(3, settings) == 3

    with pytest.raises(ConfigurationError, match="empty"):
        validate_query("   ", settings)
    with pytest.raises(ConfigurationError, match="500"):
        validate_query("x" * 501, settings)
    with pytest.raises(ConfigurationError, match="between 1 and 10"):
        validate_top_k(11, settings)
    with pytest.raises(ConfigurationError, match="integer"):
        validate_top_k(True, settings)


def test_full_mode_requires_cloud_milvus_credentials():
    with pytest.raises(ConfigurationError, match="MILVUS_ENDPOINT and MILVUS_TOKEN"):
        load_settings(require_gemini=False, env={"RAG_MODE": "full"})


def test_full_mode_can_be_loaded_without_gemini_key_for_non_generation_tools():
    settings = load_settings(
        require_gemini=False,
        env={
            "RAG_MODE": "full",
            "MILVUS_ENDPOINT": "https://cloud.example",
            "MILVUS_TOKEN": "user:secret",
        },
    )

    assert settings.gemini_api_key is None
    assert settings.milvus_endpoint == "https://cloud.example"
    assert settings.milvus_token == "user:secret"
    assert replace(settings, elasticsearch_index="custom").elasticsearch_index == "custom"
