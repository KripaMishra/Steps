import json
import torch
import logging
from elasticsearch import Elasticsearch
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize encoders and tokenizers
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Global variables
collection = None
es = None

def setup(collection_name, es_host, es_port, milvus_host, milvus_port):
    global collection, es
    try:
        connections.connect("default", host=milvus_host, port=milvus_port)
        es = Elasticsearch([{'host': es_host, 'port': es_port, 'scheme': "http"}])
        
        dim = 768  # DPR embedding dimension
        
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            ]
            schema = CollectionSchema(fields, description=collection_name)
            collection = Collection(collection_name, schema)
            
            index = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128},
            }
            collection.create_index("embedding", index)
        logging.info("Milvus and Elasticsearch setup completed.")
    except Exception as e:
        logging.error(f"Error during setup: {e}")
        raise

def encode_passage(passage):
    try:
        inputs = context_tokenizer(passage, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            embeddings = context_encoder(**inputs).pooler_output
        return embeddings[0].numpy()
    except Exception as e:
        logging.error(f"Error encoding passage: {e}")
        raise

def index_documents(documents):
    global collection, es
    try:
        ids = []
        embeddings = []
        contents = []
        
        for doc in documents:
            doc_id = doc['id']
            content = doc['content']
            
            # Index in Elasticsearch
            es.index(index="documents", id=doc_id, body={"content": content})
            
            # Prepare for Milvus indexing
            embedding = encode_passage(content)
            
            ids.append(doc_id)
            embeddings.append(embedding.tolist())
            contents.append(content)
        
        # Index in Milvus
        collection.insert([ids, embeddings, contents])
        logging.info("Documents indexed successfully.")
    except Exception as e:
        logging.error(f"Error indexing documents: {e}")
        raise

def load_data(input_path):
    try:
        with open(input_path, 'r') as content:
            documents = json.load(content)
        return documents
    except Exception as e:
        logging.error(f"Error loading data from {input_path}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data ingestion script for Elasticsearch and Milvus.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSON file containing documents.')
    parser.add_argument('--collection_name', type=str, default='Test_collection', help='Name of the Milvus collection.')
    parser.add_argument('--es_host', type=str, default='localhost', help='Elasticsearch host.')
    parser.add_argument('--es_port', type=int, default=9200, help='Elasticsearch port.')
    parser.add_argument('--milvus_host', type=str, default='localhost', help='Milvus host.')
    parser.add_argument('--milvus_port', type=int, default=19530, help='Milvus port.')

    args = parser.parse_args()

    try:
        setup(args.collection_name, args.es_host, args.es_port, args.milvus_host, args.milvus_port)
        documents = load_data(args.input_path)
        index_documents(documents)
        logging.info("Database setup and data ingestion completed.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
# Example: python components/s3_data_ingestion.py --input_path result/preprocessed_chunks.json --collection_name Test_collection --es_host localhost --es_port 9200 --milvus_host localhost --milvus_port 19530
