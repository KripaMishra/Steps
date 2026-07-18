import pytest

from components.settings import ConfigurationError, load_settings


def test_missing_gemini_key_fails_clearly():
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        load_settings(env={})


def test_settings_use_gemini_compatible_defaults():
    settings = load_settings(env={"GEMINI_API_KEY": "test-key"})

    assert settings.gemini_api_key == "test-key"
    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert settings.milvus_host == "localhost"
    assert settings.milvus_port == 19530
    assert settings.elasticsearch_host == "localhost"
    assert settings.elasticsearch_port == 9200
    assert settings.collection_name == "Test_collection"
    assert settings.elasticsearch_index == "documents"
