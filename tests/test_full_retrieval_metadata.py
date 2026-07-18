from components import s4_data_retrieval as retrieval_module
from components.settings import load_settings


class FakeCollection:
    def query(self, **kwargs):
        assert kwargs["output_fields"] == ["id", "content"]
        return [{"id": 2, "content": "Pinned memory supports asynchronous copies."}]


class FakeElasticsearch:
    def __init__(self):
        self.index = None

    def mget(self, *, index, body):
        self.index = index
        assert body == {"ids": ["2"]}
        return {
            "docs": [
                {
                    "_id": "2",
                    "found": True,
                    "_source": {
                        "source_url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
                        "title": "CUDA C++ Programming Guide",
                        "section": "Page-locked host memory",
                        "chunk_id": "memory-pinned",
                        "source_date": "2026-07-18",
                        "corpus_version": "2026-07-18",
                        "provenance": "Project-authored summary.",
                    },
                }
            ]
        }


def test_full_retrieval_uses_configured_index_and_preserves_metadata(monkeypatch):
    fake_es = FakeElasticsearch()
    monkeypatch.setattr(retrieval_module, "collection", FakeCollection())
    monkeypatch.setattr(retrieval_module, "es", fake_es)
    settings = load_settings(
        require_gemini=False,
        env={"RAG_MODE": "full", "ELASTICSEARCH_INDEX": "cuda-custom"},
    )
    retriever = retrieval_module.CustomRetrieval(settings=settings)

    records = retriever.fetch_document_records([2])

    assert fake_es.index == "cuda-custom"
    assert records == [
        {
            "document_id": "2",
            "chunk_id": "memory-pinned",
            "document": "Pinned memory supports asynchronous copies.",
            "source_url": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/",
            "title": "CUDA C++ Programming Guide",
            "section": "Page-locked host memory",
            "source_date": "2026-07-18",
            "corpus_version": "2026-07-18",
            "provenance": "Project-authored summary.",
        }
    ]
