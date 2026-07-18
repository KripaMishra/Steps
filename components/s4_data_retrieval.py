"""Existing Milvus + Elasticsearch hybrid retrieval used only in full mode."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

from components.settings import load_settings, validate_query, validate_top_k


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

collection = None
es = None
_question_encoder = None
_question_tokenizer = None
_t5_tokenizer = None
_t5_model = None


def _question_components():
    global _question_encoder, _question_tokenizer
    if _question_encoder is None or _question_tokenizer is None:
        from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

        model = "facebook/dpr-question_encoder-single-nq-base"
        _question_encoder = DPRQuestionEncoder.from_pretrained(model)
        _question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model)
    return _question_encoder, _question_tokenizer


def _expansion_components():
    global _t5_tokenizer, _t5_model
    if _t5_tokenizer is None or _t5_model is None:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        _t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
        _t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")
    return _t5_model, _t5_tokenizer


class CustomRetrieval:
    """Preserves the original full-mode hybrid search and ranking path."""

    def __init__(self, settings=None):
        self.settings = settings or load_settings(require_gemini=False)
        self.es_index = self.settings.elasticsearch_index

    def setup(
        self,
        *,
        milvus_host=None,
        milvus_port=None,
        es_host=None,
        es_port=None,
        collection_name=None,
        es_index=None,
    ):
        global collection, es
        from elasticsearch import Elasticsearch
        from pymilvus import Collection, connections

        milvus_host = milvus_host or self.settings.milvus_host
        milvus_port = milvus_port or self.settings.milvus_port
        es_host = es_host or self.settings.elasticsearch_host
        es_port = es_port or self.settings.elasticsearch_port
        collection_name = collection_name or self.settings.collection_name
        self.es_index = es_index or self.settings.elasticsearch_index
        connections.connect("default", host=milvus_host, port=milvus_port)
        es = Elasticsearch([{"host": es_host, "port": es_port, "scheme": "http"}])
        collection = Collection(collection_name)
        collection.load()
        logger.info("Setup complete: Elasticsearch and Milvus collection initialized")

    def encode_query(self, query):
        import torch

        encoder, tokenizer = _question_components()
        try:
            input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                embeddings = encoder(input_ids).pooler_output
            return embeddings[0].numpy()
        except Exception as exc:
            logger.error("Query encoding failed: %s", exc)
            return None

    def hybrid_search(self, query, top_k=100, alpha=0.5):
        """Run the original BM25 + DPR weighted score combination unchanged."""

        global collection, es
        try:
            es_response = es.search(
                index=self.es_index,
                body={"query": {"match": {"content": query}}, "size": top_k},
            )
            bm25_results = [
                (hit["_id"], hit["_score"]) for hit in es_response["hits"]["hits"]
            ]

            query_vector = self.encode_query(query)
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
                "offset": 0,
                "limit": top_k,
                "with_distance": True,
                "expr": None,
                "output_fields": ["id", "content"],
                "round_decimal": -1,
                "rerank": {
                    "metric_type": "IP",
                    "params": {"rerank_topk": min(top_k * 2, 100)},
                },
            }
            milvus_results = collection.search(
                data=[query_vector.tolist()],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
            )
            dpr_results = [
                (hit.entity.get("id"), hit.score) for hit in milvus_results[0]
            ]

            all_ids = {document_id for document_id, _ in bm25_results + dpr_results}
            combined_scores = {}
            for document_id in all_ids:
                bm25_score = next(
                    (
                        score
                        for candidate_id, score in bm25_results
                        if candidate_id == document_id
                    ),
                    0,
                )
                dpr_score = next(
                    (
                        score
                        for candidate_id, score in dpr_results
                        if candidate_id == document_id
                    ),
                    0,
                )
                combined_scores[document_id] = (
                    alpha * bm25_score + (1 - alpha) * dpr_score
                )

            return sorted(
                combined_scores.items(), key=lambda item: item[1], reverse=True
            )[:top_k]
        except Exception as exc:
            logger.error("Hybrid search failed: %s", exc)
            return []

    def _milvus_documents(self, doc_ids) -> dict[str, str]:
        global collection
        valid_ids = [int(document_id) for document_id in doc_ids if document_id is not None]
        if not valid_ids:
            return {}
        results = collection.query(
            expr=f"id in {valid_ids}", output_fields=["id", "content"]
        )
        return {str(result["id"]): result["content"] for result in results}

    def fetch_documents(self, doc_ids):
        """Compatibility helper returning document text in requested ID order."""

        try:
            documents = self._milvus_documents(doc_ids)
            return [documents[str(document_id)] for document_id in doc_ids if str(document_id) in documents]
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
            source = metadata.get(key, {})
            records.append(
                {
                    "document_id": key,
                    "chunk_id": str(source.get("chunk_id", key)),
                    "document": documents[key],
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
        try:
            model, tokenizer = _expansion_components()
            input_ids = tokenizer(
                f"expand query: {query}", return_tensors="pt"
            ).input_ids
            outputs = model.generate(
                input_ids,
                max_length=50,
                num_return_sequences=num_expansions,
                num_beams=num_expansions,
                temperature=0.7,
            )
            expanded_queries = [
                tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]

            keyword_ids = tokenizer(
                f"generate keywords for: {query}", return_tensors="pt"
            ).input_ids
            keyword_outputs = model.generate(
                keyword_ids,
                max_length=30,
                num_return_sequences=1,
                num_beams=num_keywords,
                temperature=0.7,
            )
            keywords = tokenizer.decode(
                keyword_outputs[0], skip_special_tokens=True
            ).split()
            return [
                f"{expanded_query} {' '.join(keywords)}"
                for expanded_query in [query] + expanded_queries
            ]
        except Exception as exc:
            logger.error("Query expansion failed: %s", exc)
            return []

    def retrieve_and_rerank(self, query, top_k=3):
        query = validate_query(query, self.settings)
        top_k = validate_top_k(top_k, self.settings)
        try:
            expanded_queries = self.expand_query_with_keywords(query)
            all_results = []
            for expanded_query in expanded_queries:
                all_results.extend(
                    self.hybrid_search(expanded_query, top_k=top_k * 2)
                )

            unique_results = list(dict.fromkeys(all_results))[:top_k]
            records = {
                record["document_id"]: record
                for record in self.fetch_document_records(
                    [document_id for document_id, _ in unique_results]
                )
            }
            results = []
            for document_id, score in unique_results:
                record = records.get(str(document_id))
                if record:
                    results.append({**record, "score": float(score)})

            return {
                "timestamp": datetime.now().isoformat(),
                "original_query": query,
                "expanded_queries": expanded_queries,
                "results": results,
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


class QueryEnhancer:
    def __init__(
        self,
        model_name: str = "t5-small",
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.logger = logging.getLogger(__name__)

    def expand_query_with_keywords(
        self, query: str, num_expansions: int = 2, num_keywords: int = 5
    ) -> List[str]:
        try:
            expanded_queries = self._generate_expanded_queries(query, num_expansions)
            keywords = self._generate_keywords(query, num_keywords)
            return [
                f"{candidate} {' '.join(keywords)}"
                for candidate in [query] + expanded_queries
            ]
        except Exception as exc:
            self.logger.error("Query enhancement failed: %s", exc)
            return [query]

    def _generate_expanded_queries(self, query: str, num_expansions: int) -> List[str]:
        input_ids = self.tokenizer(
            f"Give {num_expansions} alternate ways to write this query: {query}",
            return_tensors="pt",
        ).input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=num_expansions,
            num_beams=num_expansions,
            temperature=0.3,
            do_sample=True,
        )
        return [
            self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs
        ]

    def _generate_keywords(self, query: str, num_keywords: int) -> List[str]:
        keyword_ids = self.tokenizer(
            f"Add {num_keywords} keywords that summarize this query: {query}",
            return_tensors="pt",
        ).input_ids.to(self.device)
        outputs = self.model.generate(
            keyword_ids,
            max_length=30,
            num_return_sequences=1,
            num_beams=num_keywords,
            temperature=0.6,
            do_sample=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split()

    def enhance_query(
        self,
        query: str,
        num_expansions: int = 2,
        num_keywords: int = 3,
        max_length: Optional[int] = None,
    ) -> List[str]:
        enhanced_queries = self.expand_query_with_keywords(
            query, num_expansions, num_keywords
        )
        if max_length:
            return [candidate[:max_length] for candidate in enhanced_queries]
        return enhanced_queries


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
