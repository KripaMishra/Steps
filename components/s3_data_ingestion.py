"""Ingest CUDA chunks into the existing Milvus + Elasticsearch full-mode stores."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from components.settings import Settings, load_settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

collection = None
es = None
elasticsearch_index = "documents"
_context_encoder = None
_context_tokenizer = None


def setup(
    collection_name: str | None = None,
    es_host: str | None = None,
    es_port: int | None = None,
    milvus_host: str | None = None,
    milvus_port: int | None = None,
    *,
    es_index: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Connect full-mode stores without changing the existing Milvus schema."""

    global collection, es, elasticsearch_index
    settings = settings or load_settings(require_gemini=False)
    collection_name = collection_name or settings.collection_name
    es_host = es_host or settings.elasticsearch_host
    es_port = es_port or settings.elasticsearch_port
    milvus_host = milvus_host or settings.milvus_host
    milvus_port = milvus_port or settings.milvus_port
    elasticsearch_index = es_index or settings.elasticsearch_index

    from elasticsearch import Elasticsearch
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    connections.connect("default", host=milvus_host, port=milvus_port)
    es = Elasticsearch([{"host": es_host, "port": es_port, "scheme": "http"}])

    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        ]
        collection = Collection(
            collection_name,
            CollectionSchema(fields, description=collection_name),
        )
        collection.create_index(
            "embedding",
            {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            },
        )
    logger.info("Milvus and Elasticsearch setup completed")


def _get_context_encoder():
    global _context_encoder, _context_tokenizer
    if _context_encoder is None or _context_tokenizer is None:
        from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

        model = "facebook/dpr-ctx_encoder-single-nq-base"
        _context_encoder = DPRContextEncoder.from_pretrained(model)
        _context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(model)
    return _context_encoder, _context_tokenizer


def encode_passage(passage: str):
    import torch

    encoder, tokenizer = _get_context_encoder()
    inputs = tokenizer(
        passage,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        embeddings = encoder(**inputs).pooler_output
    return embeddings[0].numpy()


def elasticsearch_document(document: dict[str, Any]) -> dict[str, Any]:
    """Build the additive Elasticsearch source while keeping content compatible."""

    document_id = document["id"]
    return {
        "content": document["content"],
        "source_url": document.get("source_url", document.get("url", "")),
        "title": document.get("title", ""),
        "section": document.get("section", ""),
        "chunk_id": str(document.get("chunk_id", document_id)),
        "source_date": document.get("source_date", ""),
        "corpus_version": document.get("corpus_version", ""),
        "provenance": document.get("provenance", ""),
    }


def index_documents(documents: list[dict[str, Any]], *, index_name: str | None = None) -> None:
    if collection is None or es is None:
        raise RuntimeError("setup() must be called before index_documents()")

    ids = []
    embeddings = []
    contents = []
    target_index = index_name or elasticsearch_index

    for document in documents:
        document_id = document["id"]
        content = document["content"]
        es.index(
            index=target_index,
            id=document_id,
            document=elasticsearch_document(document),
        )
        ids.append(document_id)
        embeddings.append(list(encode_passage(content)))
        contents.append(content)

    collection.insert([ids, embeddings, contents])
    logger.info("Indexed %d documents", len(documents))


def load_data(input_path: str | Path) -> list[dict[str, Any]]:
    with Path(input_path).open(encoding="utf-8") as content:
        payload = json.load(content)
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return payload["records"]
    if not isinstance(payload, list):
        raise ValueError("ingestion input must be a list or a fixture records object")
    return payload


def main() -> None:
    settings = load_settings(require_gemini=False)
    parser = argparse.ArgumentParser(
        description="Ingest data into the full Milvus + Elasticsearch backend"
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--collection_name", default=settings.collection_name)
    parser.add_argument("--es_host", default=settings.elasticsearch_host)
    parser.add_argument("--es_port", type=int, default=settings.elasticsearch_port)
    parser.add_argument("--es_index", default=settings.elasticsearch_index)
    parser.add_argument("--milvus_host", default=settings.milvus_host)
    parser.add_argument("--milvus_port", type=int, default=settings.milvus_port)
    args = parser.parse_args()

    setup(
        args.collection_name,
        args.es_host,
        args.es_port,
        args.milvus_host,
        args.milvus_port,
        es_index=args.es_index,
        settings=settings,
    )
    index_documents(load_data(args.input_path))


if __name__ == "__main__":
    main()
