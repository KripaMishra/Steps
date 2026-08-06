"""Milvus + Elasticsearch hybrid retrieval used only in full mode."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from components.settings import (
    ConfigurationError,
    load_settings,
    validate_query,
    validate_top_k,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

collection = None
es = None


def _rrf_fuse(
    lexical_results: list[tuple[str, float]],
    dense_results: list[tuple[str, float]],
    *,
    lexical_weight: float = 0.5,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    """Combine rankings without pretending BM25 and vector scores share a scale."""

    scores: dict[str, float] = {}
    for weight, ranked in (
        (lexical_weight, lexical_results),
        (1.0 - lexical_weight, dense_results),
    ):
        for rank, (document_id, _score) in enumerate(ranked, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + weight / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class CustomRetrieval:
    """Milvus retrieval with optional Elasticsearch rank fusion."""

    def __init__(self, settings=None):
        self.settings = settings or load_settings(require_gemini=False)
        self.collection_name = self.settings.collection_name
        self.es_index = self.settings.elasticsearch_index

    def setup(
        self,
        *,
        milvus_endpoint=None,
        milvus_token=None,
        es_host=None,
        es_port=None,
        collection_name=None,
        es_index=None,
    ):
        global collection, es
        from pymilvus import MilvusClient

        milvus_endpoint = milvus_endpoint or self.settings.milvus_endpoint
        milvus_token = milvus_token or self.settings.milvus_token
        if not milvus_endpoint or not milvus_token:
            raise ConfigurationError(
                "MILVUS_ENDPOINT and MILVUS_TOKEN are required for Milvus setup"
            )
        es_host = es_host or self.settings.elasticsearch_host
        es_port = es_port or self.settings.elasticsearch_port
        collection_name = collection_name or self.settings.collection_name
        self.collection_name = collection_name
        self.es_index = es_index or self.settings.elasticsearch_index
        collection = MilvusClient(uri=milvus_endpoint, token=milvus_token)
        es = None
        if self.settings.elasticsearch_enabled:
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
                logger.warning("Elasticsearch unavailable; using Milvus only: %s", exc)
        collection.load_collection(collection_name)
        logger.info(
            "Setup complete: Milvus collection initialized%s",
            " with Elasticsearch" if es is not None else " without Elasticsearch",
        )

    def encode_query(self, query) -> list[float] | None:
        try:
            from components.gemini_embeddings import GeminiEmbedder

            if not hasattr(self, "_embedder"):
                self._embedder = GeminiEmbedder(self.settings)
            return self._embedder.embed_query(query)
        except Exception as exc:
            logger.error("Query encoding failed: %s", exc)
            return None

    def hybrid_search(self, query, top_k=100, alpha=0.5):
        """Fuse lexical and dense rankings with weighted reciprocal rank fusion."""

        global collection, es
        try:
            bm25_results = []
            if es is not None:
                try:
                    es_response = es.search(
                        index=self.es_index,
                        body={"query": {"match": {"content": query}}, "size": top_k},
                    )
                    bm25_results = [
                        (str(hit["_id"]), hit.get("_score", 0.0))
                        for hit in es_response["hits"]["hits"]
                    ]
                except Exception as exc:
                    logger.warning("Elasticsearch search failed; using Milvus only: %s", exc)
                    es = None

            query_vector = self.encode_query(query)
            if query_vector is None:
                return []
            milvus_results = collection.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="embedding",
                search_params={"metric_type": "IP", "params": {"nprobe": 16}},
                limit=top_k,
                output_fields=["id", "content"],
            )
            dense_results = [
                (str(hit["id"]), hit["distance"]) for hit in milvus_results[0]
            ]
            if not bm25_results:
                return dense_results[:top_k]
            return _rrf_fuse(bm25_results, dense_results, lexical_weight=alpha)[:top_k]
        except Exception as exc:
            logger.error("Hybrid search failed: %s", exc)
            return []

    def _milvus_documents(self, doc_ids) -> dict[str, dict[str, Any]]:
        global collection
        valid_ids = [int(document_id) for document_id in doc_ids if document_id is not None]
        if not valid_ids:
            return {}
        results = collection.query(
            collection_name=self.collection_name,
            filter=f"id in {valid_ids}",
            output_fields=["id", "content", "metadata"],
        )
        return {
            str(result["id"]): {
                "content": result["content"],
                "metadata": result.get("metadata", {}),
            }
            for result in results
        }

    def fetch_documents(self, doc_ids):
        """Compatibility helper returning document text in requested ID order."""

        try:
            documents = self._milvus_documents(doc_ids)
            return [
                documents[str(document_id)]["content"]
                for document_id in doc_ids
                if str(document_id) in documents
            ]
        except Exception as exc:
            logger.error("Fetching documents failed: %s", exc)
            return []

    def fetch_document_records(self, doc_ids) -> list[dict[str, Any]]:
        """Merge additive Elasticsearch provenance with Milvus document text."""

        global es
        try:
            documents = self._milvus_documents(doc_ids)
        except Exception as exc:
            logger.error("Fetching documents failed: %s", exc)
            return []

        metadata: dict[str, dict[str, Any]] = {}
        if es is not None and documents:
            try:
                response = es.mget(
                    index=self.es_index,
                    body={"ids": list(documents)},
                )
                metadata = {
                    str(item["_id"]): item.get("_source", {})
                    for item in response.get("docs", [])
                    if item.get("found", True)
                }
            except Exception as exc:
                logger.warning("Source metadata lookup failed: %s", exc)

        records = []
        for document_id in doc_ids:
            key = str(document_id)
            if key not in documents:
                continue
            milvus_document = documents[key]
            source = {
                **milvus_document["metadata"],
                **metadata.get(key, {}),
            }
            records.append(
                {
                    "document_id": key,
                    "chunk_id": str(source.get("chunk_id", key)),
                    "document": milvus_document["content"],
                    "source_url": source.get("source_url", ""),
                    "title": source.get("title", ""),
                    "section": source.get("section", ""),
                    "source_date": source.get("source_date", ""),
                    "corpus_version": source.get("corpus_version", ""),
                    "provenance": source.get("provenance", ""),
                }
            )
        return records

    def expand_query_with_keywords(self, query, num_expansions=3, num_keywords=5):
        """Compatibility hook; Gemini embeddings search the original query directly."""

        del num_expansions, num_keywords
        return [query]

    def retrieve_and_rerank(self, query, top_k=3):
        query = validate_query(query, self.settings)
        top_k = validate_top_k(top_k, self.settings)
        try:
            expanded_queries = [query]

            candidate_limit = max(top_k * 5, 20)
            candidate_scores: dict[str, float] = {}
            for expanded_query in expanded_queries:
                for document_id, score in self.hybrid_search(
                    expanded_query, top_k=candidate_limit
                ):
                    key = str(document_id)
                    candidate_scores[key] = candidate_scores.get(key, 0.0) + float(score)

            ranked_candidates = sorted(
                candidate_scores.items(), key=lambda item: (-item[1], item[0])
            )
            records = {
                record["document_id"]: record
                for record in self.fetch_document_records(
                    [document_id for document_id, _ in ranked_candidates]
                )
            }
            candidates = [
                {**records[document_id], "score": score}
                for document_id, score in ranked_candidates
                if document_id in records
            ]

            retrieval_strategy = (
                "rrf+gemini-embeddings" if es is not None else "gemini-embeddings"
            )
            return {
                "timestamp": datetime.now().isoformat(),
                "original_query": query,
                "expanded_queries": expanded_queries,
                "candidate_count": len(candidates),
                "retrieval_strategy": retrieval_strategy,
                "results": candidates[:top_k],
            }
        except Exception as exc:
            logger.error("Retrieve and rerank failed: %s", exc)
            return {}

    def save_query_results(self, query_data, file_path=None):
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.settings.result_dir / f"query_results_{timestamp}.json"
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as output:
            json.dump(query_data, output, ensure_ascii=False, indent=4)
        logger.info("Query results saved to %s", file_path)


def main(query, top_k, file_path):
    retrieval = CustomRetrieval()
    retrieval.setup()
    results = retrieval.retrieve_and_rerank(query, top_k)
    retrieval.save_query_results(results, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Retrieval and Reranking")
    parser.add_argument("query", help="The query string")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--file_path", default=None)
    args = parser.parse_args()
    main(args.query, args.top_k, args.file_path)
