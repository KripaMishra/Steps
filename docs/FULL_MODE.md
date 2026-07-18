# Full Milvus + Elasticsearch mode

Full mode preserves the original hybrid retrieval architecture and is intended for advanced local self-hosting. It is not the default public demo.

## Requirements

- Python 3.11+
- dependencies from `requirements.txt`
- Milvus and Elasticsearch
- disk/memory for DPR and T5 model downloads
- a valid `GEMINI_API_KEY`
- a corpus that you are permitted to crawl, store, and process

Review [DATA_PROVENANCE.md](../DATA_PROVENANCE.md) first. NVIDIA site terms, documentation licenses, and crawl policy can change. Generated crawl output is local operator data and must not be committed without a separate rights review.

## Start services

```bash
cp .env.example .env
# Set GEMINI_API_KEY in .env
docker compose --profile full up -d milvus elasticsearch
```

The Compose databases are reachable only on the internal network. For host-side ingestion, either run ingestion in a container attached to that network or deliberately publish local-only ports in an uncommitted override file. Do not expose database ports publicly.

The historical `standalone_embed.sh` remains available for a host-local Milvus instance. Elasticsearch must also be started separately when using that helper.

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

The legacy cleaner/chunker produces content-only chunks. Citation-backed public use requires enriching every chunk before ingestion with:

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
  --es_host "$ELASTICSEARCH_HOST" --es_port "$ELASTICSEARCH_PORT" \
  --milvus_host "$MILVUS_HOST" --milvus_port "$MILVUS_PORT"
```

Ingestion keeps the Milvus schema unchanged. The configured Elasticsearch index stores `content` plus additive source metadata. Retrieval uses that same configured index and merges its metadata with Milvus content after ranking.

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
5. Ingest into the configured index and collection.
6. Run focused source checks and the offline suite.
7. Keep raw or processed source data out of Git unless redistribution rights are documented.
