import sys
import types

import components.s3_data_ingestion as ingestion
from components.settings import load_settings


class FakeElasticsearch:
    def __init__(self):
        self.calls = []

    def index(self, **kwargs):
        self.calls.append(kwargs)


class FakeCollection:
    def __init__(self):
        self.inserted = None

    def insert(self, collection_name, *, data):
        self.inserted = (collection_name, data)


def test_ingestion_continues_without_elasticsearch(monkeypatch):
    fake_collection = FakeCollection()
    monkeypatch.setattr(ingestion, "es", None)
    monkeypatch.setattr(ingestion, "collection", fake_collection)
    monkeypatch.setattr(ingestion, "encode_passage", lambda content: [0.1, 0.2])

    ingestion.index_documents([{"id": 7, "content": "CUDA streams order work."}])

    assert fake_collection.inserted == (
        "Test_collection",
        [
            {
                "id": 7,
                "embedding": [0.1, 0.2],
                "content": "CUDA streams order work.",
                "metadata": {
                    "source_url": "",
                    "title": "",
                    "section": "",
                    "chunk_id": "7",
                    "source_date": "",
                    "corpus_version": "",
                    "provenance": "",
                },
            }
        ],
    )


def test_ingestion_uses_configured_index_and_preserves_source_metadata(monkeypatch):
    fake_es = FakeElasticsearch()
    fake_collection = FakeCollection()
    monkeypatch.setattr(ingestion, "es", fake_es)
    monkeypatch.setattr(ingestion, "collection", fake_collection)
    monkeypatch.setattr(ingestion, "elasticsearch_index", "cuda-custom")
    monkeypatch.setattr(ingestion, "encode_passage", lambda content: [0.1, 0.2])

    ingestion.index_documents(
        [
            {
                "id": 7,
                "chunk_id": "streams-7",
                "content": "CUDA streams order work.",
                "source_url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
                "title": "CUDA C++ Programming Guide",
                "section": "Streams",
                "source_date": "2026-07-18",
                "corpus_version": "2026-07-18",
                "provenance": "Project-authored summary.",
            }
        ]
    )

    indexed = fake_es.calls[0]
    assert indexed["index"] == "cuda-custom"
    assert indexed["id"] == 7
    assert indexed["document"] == {
        "content": "CUDA streams order work.",
        "source_url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
        "title": "CUDA C++ Programming Guide",
        "section": "Streams",
        "chunk_id": "streams-7",
        "source_date": "2026-07-18",
        "corpus_version": "2026-07-18",
        "provenance": "Project-authored summary.",
    }
    assert fake_collection.inserted == (
        "Test_collection",
        [
            {
                "id": 7,
                "embedding": [0.1, 0.2],
                "content": "CUDA streams order work.",
                "metadata": {
                    "source_url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
                    "title": "CUDA C++ Programming Guide",
                    "section": "Streams",
                    "chunk_id": "streams-7",
                    "source_date": "2026-07-18",
                    "corpus_version": "2026-07-18",
                    "provenance": "Project-authored summary.",
                },
            }
        ],
    )


def test_demo_fixture_can_be_loaded_for_full_mode_wiring_check():
    records = ingestion.load_data("demo/fixtures/cuda_docs.json")

    assert len(records) == 12
    assert records[0]["source_url"].startswith("https://docs.nvidia.com/")


def test_setup_uses_milvus_cloud_endpoint_and_token(monkeypatch):
    client_calls = []

    class FakeMilvusClient:
        def __init__(self, *, uri, token):
            client_calls.append({"uri": uri, "token": token})

        def has_collection(self, collection_name):
            return True

        def describe_collection(self, collection_name):
            return {
                "schema": {
                    "fields": [
                        {"name": "id"},
                        {"name": "embedding"},
                        {"name": "content"},
                        {"name": "metadata"},
                    ]
                }
            }

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.MilvusClient = FakeMilvusClient
    pymilvus.DataType = object()
    monkeypatch.setitem(sys.modules, "pymilvus", pymilvus)

    ingestion.setup(
        settings=load_settings(
            env={
                "MILVUS_ENDPOINT": "https://cloud.example",
                "MILVUS_TOKEN": "token-value",
            }
        )
    )

    assert client_calls == [{"uri": "https://cloud.example", "token": "token-value"}]


def test_elasticsearch_document_accepts_legacy_url_and_defaults_chunk_id():
    document = ingestion.elasticsearch_document(
        {"id": 3, "content": "text", "url": "https://example.com", "title": "Title"}
    )

    assert document["source_url"] == "https://example.com"
    assert document["chunk_id"] == "3"
    assert document["section"] == ""
