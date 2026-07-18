import components.s3_data_ingestion as ingestion


class FakeElasticsearch:
    def __init__(self):
        self.calls = []

    def index(self, **kwargs):
        self.calls.append(kwargs)


class FakeCollection:
    def __init__(self):
        self.inserted = None

    def insert(self, rows):
        self.inserted = rows


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
    assert fake_collection.inserted == [[7], [[0.1, 0.2]], ["CUDA streams order work."]]


def test_demo_fixture_can_be_loaded_for_full_mode_wiring_check():
    records = ingestion.load_data("demo/fixtures/cuda_docs.json")

    assert len(records) == 12
    assert records[0]["source_url"].startswith("https://docs.nvidia.com/")


def test_elasticsearch_document_accepts_legacy_url_and_defaults_chunk_id():
    document = ingestion.elasticsearch_document(
        {"id": 3, "content": "text", "url": "https://example.com", "title": "Title"}
    )

    assert document["source_url"] == "https://example.com"
    assert document["chunk_id"] == "3"
    assert document["section"] == ""
