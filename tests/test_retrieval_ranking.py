import types

from components import s4_data_retrieval as retrieval_module
from components.s4_data_retrieval import _rrf_fuse
from components.settings import load_settings


def test_rrf_fusion_uses_rank_not_incomparable_backend_scores():
    result = _rrf_fuse(
        [("lexical-top", 100.0), ("shared", 1.0)],
        [("shared", 0.99), ("dense-top", 0.1)],
    )

    assert [document_id for document_id, _ in result] == [
        "shared",
        "lexical-top",
        "dense-top",
    ]


def test_rrf_fusion_is_deterministic_for_ties():
    result = _rrf_fuse([("b", 1.0)], [("a", 1.0)])

    assert [document_id for document_id, _ in result] == ["a", "b"]


def test_hybrid_search_falls_back_to_milvus_without_elasticsearch(monkeypatch):
    hit = {"id": 7, "distance": 0.9}
    monkeypatch.setattr(
        retrieval_module,
        "collection",
        types.SimpleNamespace(search=lambda **kwargs: [[hit]]),
    )
    monkeypatch.setattr(retrieval_module, "es", None)
    retriever = retrieval_module.CustomRetrieval(
        settings=load_settings(
            require_gemini=False,
            env={
                "MILVUS_ENDPOINT": "https://cloud.example",
                "MILVUS_TOKEN": "token-value",
            },
        )
    )
    monkeypatch.setattr(retriever, "encode_query", lambda query: [0.1, 0.2])

    assert retriever.hybrid_search("CUDA streams", top_k=3) == [("7", 0.9)]
