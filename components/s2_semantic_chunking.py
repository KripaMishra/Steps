import os
import json
import logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logging.error(f"Error loading text from {file_path}: {e}")
        raise

def split_into_sentences(text):
    try:
        return [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    except Exception as e:
        logging.error(f"Error splitting text into sentences: {e}")
        raise

def split_large_chunk(chunk, max_length=300):
    try:
        words = chunk.split()
        sub_chunks = []
        current_sub_chunk = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 > max_length:
                sub_chunks.append(' '.join(current_sub_chunk))
                current_sub_chunk = [word]
                current_length = len(word)
            else:
                current_sub_chunk.append(word)
                current_length += len(word) + 1

        if current_sub_chunk:
            sub_chunks.append(' '.join(current_sub_chunk))

        return sub_chunks
    except Exception as e:
        logging.error(f"Error splitting large chunk: {e}")
        raise

def semantic_chunking(text, threshold=0.3, max_chunk_length=300):
    try:
        sentences = split_into_sentences(text)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        chunks = []
        current_chunk = [0]
        
        for i in range(1, len(sentences)):
            if np.max(similarity_matrix[i, current_chunk]) >= threshold:
                current_chunk.append(i)
            else:
                chunks.append(current_chunk)
                current_chunk = [i]
        
        if current_chunk:
            chunks.append(current_chunk)
        
        result = []
        for chunk in chunks:
            chunk_text = ' '.join([sentences[idx] for idx in chunk])
            if len(chunk_text) > max_chunk_length:
                result.extend(split_large_chunk(chunk_text, max_chunk_length))
            else:
                result.append(chunk_text)
        
        return result
    except Exception as e:
        logging.error(f"Error in semantic chunking: {e}")
        raise

def save_chunks_to_json(chunks, output_file, micro_file, micro_threshold=100):
    try:
        regular_chunks = []
        micro_chunks = []
        chunk_id = 1
        
        for chunk in chunks:
            chunk_dict = {"id": chunk_id, "content": chunk}
            if len(chunk) < micro_threshold:
                micro_chunks.append(chunk_dict)
            else:
                regular_chunks.append(chunk_dict)
            chunk_id += 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(regular_chunks, f, ensure_ascii=False, indent=2)
        
        with open(micro_file, 'w', encoding='utf-8') as f:
            json.dump(micro_chunks, f, ensure_ascii=False, indent=2)
        
        return len(regular_chunks), len(micro_chunks)
    except Exception as e:
        logging.error(f"Error saving chunks to JSON: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic chunking script.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input text file.')
    parser.add_argument('--similarity_threshold', type=float, default=0.15, help='Similarity threshold for chunking.')
    parser.add_argument('--max_chunk_length', type=int, default=400, help='Maximum length of each chunk.')
    parser.add_argument('--output_json_file', type=str, required=True, help='Path to save the regular chunks JSON file.')
    parser.add_argument('--micro_json_file', type=str, required=True, help='Path to save the micro chunks JSON file.')
    parser.add_argument('--micro_threshold', type=int, default=100, help='Threshold for micro chunks.')

    args = parser.parse_args()

    try:
        text = load_text(args.file_path)
        chunks = semantic_chunking(text, args.similarity_threshold, args.max_chunk_length)

        logging.info(f"Total number of chunks: {len(chunks)}")

        # Save chunks to JSON files
        regular_count, micro_count = save_chunks_to_json(chunks, args.output_json_file, args.micro_json_file, args.micro_threshold)

        logging.info(f"Regular chunks saved to {args.output_json_file}: {regular_count}")
        logging.info(f"Micro chunks saved to {args.micro_json_file}: {micro_count}")

        # Print some example chunks
        logging.info("\nExample chunks:")
        for i, chunk in enumerate(chunks[:5], 1):  # Print first 5 chunks as examples
            logging.info(f"\nChunk {i}:")
            logging.info(f"Content: {chunk}")
            logging.info(f"Length: {len(chunk)}")

        # Print example of JSON format
        logging.info("\nExample of JSON format:")
        example_chunks = [{"id": i, "content": chunk} for i, chunk in enumerate(chunks[:2], 1)]
        logging.info(json.dumps(example_chunks, indent=2))

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
# Example: python components/s2_semantic_chunking.py --file_path result/cleaned_data.txt --similarity_threshold 0.15 --max_chunk_length 400 --output_json_file result/preprocessed_chunks.json --micro_json_file result/micro_chunks.json --micro_threshold 100
