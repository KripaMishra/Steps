# Full Milvus mode with optional Elasticsearch

Full mode preserves the original hybrid retrieval architecture and uses a Milvus/Zilliz Cloud instance. It is not the default public demo.

## Requirements

- Python 3.11+
- dependencies from `requirements.txt`
- a Milvus/Zilliz Cloud endpoint and token
- Elasticsearch (optional; adds BM25 hybrid retrieval and provenance metadata)
- Gemini API access for embedding and answer generation
- a valid `GEMINI_API_KEY`
- a corpus that you are permitted to crawl, store, and process

Review [DATA_PROVENANCE.md](../DATA_PROVENANCE.md) first. NVIDIA site terms, documentation licenses, and crawl policy can change. Generated crawl output is local operator data and must not be committed without a separate rights review.

## Start services

```bash
cp .env.example .env
# Set GEMINI_API_KEY, MILVUS_ENDPOINT, and MILVUS_TOKEN in .env
docker compose --profile elasticsearch up -d elasticsearch
# Set ELASTICSEARCH_ENABLED=true in .env before starting app-full.
```

Elasticsearch runs only on the internal Compose network. It is optional: when it is unavailable, ingestion and retrieval continue with Milvus/Gemini vector search and use provenance metadata stored with each Milvus record; only BM25 fusion is skipped. Milvus is accessed through the configured cloud endpoint; keep its token in an environment/secret manager and never commit or log it.

## Crawl and preprocess

```bash
cd nvidia_docs
scrapy crawl nvidia_docs -O nvidia_docs/spiders/output.json
cd ..
python components/s1_data_cleaning.py \
  --file_path nvidia_docs/nvidia_docs/spiders/output.json \
  --output_path result/cleaned_data.txt
python components/s2_semantic_chunking.py \
  --file_path result/cleaned_data.txt \
  --similarity_threshold 0.15 \
  --max_chunk_length 400 \
  --output_json_file result/preprocessed_chunks.json \
  --micro_json_file result/micro_chunks.json \
  --micro_threshold 100
```

The legacy cleaner/chunker produces content-only chunks. For current answer quality, treat chunking as an evaluated retrieval parameter: preserve section boundaries, keep chunks focused, and retain enough overlap to avoid splitting definitions from their qualifiers. Citation-backed public use also requires enriching every chunk before ingestion with:

```text
id, content, source_url, title, section, chunk_id,
source_date, corpus_version, provenance
```

Do not invent missing provenance. The public fixture already has this shape and can be used for a wiring check with `make ingest-demo` once host-local services are available.

## Ingest

```bash
set -a; source .env; set +a
RAG_MODE=full python -m components.s3_data_ingestion \
  --input_path result/provenance_enriched_chunks.json \
  --collection_name "$MILVUS_COLLECTION" \
  --es_index "$ELASTICSEARCH_INDEX" \
  --es_host "$ELASTICSEARCH_HOST" --es_port "$ELASTICSEARCH_PORT"
```

Ingestion stores provenance metadata alongside 768-dimensional `gemini-embedding-001` retrieval-document embeddings with inner-product search. Retrieval embeds each query with Gemini's retrieval-query task and uses Elasticsearch BM25 plus dense Milvus reciprocal-rank fusion when Elasticsearch is available; otherwise it uses the dense Milvus ranking directly. Changing `GEMINI_EMBEDDING_MODEL`, `GEMINI_EMBEDDING_DIMENSIONS`, or the collection schema requires recreating/reindexing the Milvus collection.

## Query

```bash
RAG_MODE=full python -m components.s5_rag_llm \
  "What is CUDA occupancy?" --top_k 3
RAG_MODE=full streamlit run main.py
```

If existing documents lack URLs/titles/chunk IDs, reingest provenance-enriched records. The shared response contract deliberately refuses to present an uncited answer.

## Refresh

1. Review source permissions and crawl scope.
2. Crawl into ignored local output.
3. clean/chunk and enrich every chunk with real provenance metadata.
4. Set a new corpus version.
5. Ingest into the configured index and collection; recreate the collection when changing embedding dimensions or model family.
6. Run focused source checks and the offline suite. Track retrieval hit rate and MRR before/after retrieval changes; add faithfulness/context-precision checks for any production corpus.
7. Keep raw or processed source data out of Git unless redistribution rights are documented.
