"""Ingest CUDA chunks into the existing Milvus + Elasticsearch full-mode stores."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from components.settings import ConfigurationError, Settings, load_settings


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

collection = None
milvus_collection_name = "Test_collection"
es = None
elasticsearch_index = "documents"
_embedding_settings = None
_embedder = None


def setup(
    collection_name: str | None = None,
    es_host: str | None = None,
    es_port: int | None = None,
    milvus_endpoint: str | None = None,
    milvus_token: str | None = None,
    *,
    es_index: str | None = None,
    settings: Settings | None = None,
    recreate_collection: bool = False,
) -> None:
    """Connect full-mode stores without changing the existing Milvus schema."""

    global collection, milvus_collection_name, es, elasticsearch_index, _embedding_settings
    settings = settings or load_settings(require_gemini=False)
    _embedding_settings = settings
    collection_name = collection_name or settings.collection_name
    es_host = es_host or settings.elasticsearch_host
    es_port = es_port or settings.elasticsearch_port
    milvus_endpoint = milvus_endpoint or settings.milvus_endpoint
    milvus_token = milvus_token or settings.milvus_token
    elasticsearch_index = es_index or settings.elasticsearch_index
    milvus_collection_name = collection_name
    if not milvus_endpoint or not milvus_token:
        raise ConfigurationError(
            "MILVUS_ENDPOINT and MILVUS_TOKEN are required for Milvus setup"
        )

    from pymilvus import DataType, MilvusClient

    collection = MilvusClient(uri=milvus_endpoint, token=milvus_token)
    es = None
    if settings.elasticsearch_enabled:
        from elasticsearch import Elasticsearch

        try:
            elasticsearch = Elasticsearch(
                [{"host": es_host, "port": es_port, "scheme": "http"}],
                max_retries=0,
                retry_on_timeout=False,
            )
            if not elasticsearch.options(request_timeout=1).ping():
                raise ConnectionError("Elasticsearch ping failed")
            es = elasticsearch
        except Exception as exc:
            logger.warning("Elasticsearch unavailable; indexing Milvus only: %s", exc)

    if collection.has_collection(collection_name):
        fields = collection.describe_collection(collection_name)["schema"]["fields"]
        field_names = {field["name"] for field in fields}
        if "metadata" not in field_names:
            if not recreate_collection:
                raise ConfigurationError(
                    f"Milvus collection {collection_name!r} lacks provenance metadata. "
                    "Re-run with recreate_collection=True after backing up its data."
                )
            collection.drop_collection(collection_name)

    if not collection.has_collection(collection_name):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="embedding",
            datatype=DataType.FLOAT_VECTOR,
            dim=settings.gemini_embedding_dimensions,
        )
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)
        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="IP",
            params={"nlist": 128},
        )
        collection.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )
    logger.info(
        "Milvus setup completed%s",
        " with Elasticsearch" if es is not None else " without Elasticsearch",
    )


def _get_embedder():
    global _embedder
    if _embedder is None:
        from components.gemini_embeddings import GeminiEmbedder

        _embedder = GeminiEmbedder(_embedding_settings or load_settings())
    return _embedder


def encode_passage(passage: str) -> list[float]:
    """Encode a passage with Gemini's retrieval-document embedding task."""

    return _get_embedder().embed_document(passage)


def document_metadata(document: dict[str, Any]) -> dict[str, Any]:
    """Build provenance stored with every Milvus vector."""

    document_id = document["id"]
    return {
        "source_url": document.get("source_url", document.get("url", "")),
        "title": document.get("title", ""),
        "section": document.get("section", ""),
        "chunk_id": str(document.get("chunk_id", document_id)),
        "source_date": document.get("source_date", ""),
        "corpus_version": document.get("corpus_version", ""),
        "provenance": document.get("provenance", ""),
    }


def elasticsearch_document(document: dict[str, Any]) -> dict[str, Any]:
    """Build the additive Elasticsearch source while keeping content compatible."""

    return {"content": document["content"], **document_metadata(document)}


def index_documents(documents: list[dict[str, Any]], *, index_name: str | None = None) -> None:
    if collection is None:
        raise RuntimeError("setup() must be called before index_documents()")

    global es
    milvus_records = []
    target_index = index_name or elasticsearch_index

    for document in documents:
        document_id = document["id"]
        content = document["content"]
        if es is not None:
            try:
                es.index(
                    index=target_index,
                    id=document_id,
                    document=elasticsearch_document(document),
                )
            except Exception as exc:
                logger.warning("Elasticsearch indexing failed; continuing with Milvus: %s", exc)
                es = None
        milvus_records.append(
            {
                "id": document_id,
                "embedding": list(encode_passage(content)),
                "content": content,
                "metadata": document_metadata(document),
            }
        )

    collection.insert(milvus_collection_name, data=milvus_records)
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
    parser.add_argument(
        "--recreate_collection",
        action="store_true",
        help="Drop and recreate the collection if its schema lacks provenance metadata.",
    )
    args = parser.parse_args()

    setup(
        args.collection_name,
        args.es_host,
        args.es_port,
        es_index=args.es_index,
        settings=settings,
        recreate_collection=args.recreate_collection,
    )
    index_documents(load_data(args.input_path))


if __name__ == "__main__":
    main()
