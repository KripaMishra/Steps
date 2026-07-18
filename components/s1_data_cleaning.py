import os
import json
import re
import logging
from nltk.tokenize import sent_tokenize
import tempfile
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, file_path, output_path):
        self.file_path = file_path
        self.output_path = output_path
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_path = self.temp_file.name

    def __del__(self):
        try:
            os.remove(self.temp_path)
            logging.info(f"Deleted temporary file {self.temp_path}")
        except OSError as e:
            logging.error(f"Error deleting temporary file {self.temp_path}: {e}")

    def get_data(self):
        try:
            logging.info(f"Attempting to read file: {self.file_path}")
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"The file {self.file_path} does not exist.")

            with open(self.file_path, 'r') as json_file:
                data = json.load(json_file)

            logging.info(f"File size: {len(json.dumps(data))} characters")

            if not isinstance(data, list):
                logging.info("Data is not a list. Converting to list.")
                data = [data]

            logging.info(f"Extracted {len(data)} items")

            with open(self.temp_path, 'w') as content_file:
                items_written = 0
                for item in data:
                    if isinstance(item, dict) and 'content' in item:
                        content = item['content']
                        content = self.clean_text(content)
                        content_file.write(content + "\n")
                        items_written += 1
                    else:
                        logging.warning(f"Skipping invalid item: {item}")

            logging.info(f"Wrote {items_written} items to temporary file")

            if items_written == 0:
                raise ValueError("No valid items were found in the data.")

            return data

        except Exception as e:
            logging.error(f"Error in get_data: {e}")
            raise

    def clean_text(self, text):
        try:
            text = re.sub(r'\\n|\n|\\r|\r|//', ' ', text)
            text = re.sub(r'([^\w\s])\1{3,}', r'\1', text)
            return text.strip()
        except Exception as e:
            logging.error(f"Error in clean_text: {e}")
            raise

    def split_sentences(self):
        try:
            with open(self.temp_path, 'r') as f:
                content = f.read()

            sentences = re.split(r'(?<=\.)\s+', content)

            formatted_sentences = []
            current_sentence = ""

            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if re.match(r'^[\d\W]+$', sentence):
                    current_sentence += " " + sentence
                else:
                    if current_sentence:
                        current_sentence += " " + sentence
                        if 90 <= len(current_sentence.strip()) <= 850:
                            formatted_sentences.append(current_sentence.strip())
                        current_sentence = ""
                    else:
                        if i + 1 < len(sentences) and re.match(r'^[\d\W]+$', sentences[i + 1].strip()):
                            current_sentence = sentence
                        else:
                            if 90 <= len(sentence.strip()) <= 850:
                                formatted_sentences.append(sentence.strip())

            if current_sentence and 90 <= len(current_sentence.strip()) <= 850:
                formatted_sentences.append(current_sentence.strip())

            with open(self.output_path, 'w') as f:
                for sentence in formatted_sentences:
                    f.write(sentence + '\n')
                logging.info(f'The content file saved at {self.output_path}')

        except Exception as e:
            logging.error(f"Error in split_sentences: {e}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data cleaning script.')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the cleaned data.')

    args = parser.parse_args()

    processor = DataProcessor(args.file_path, args.output_path)
    processor.get_data()
    processor.split_sentences()


# Example: python components/s1_data_cleaning.py --file_path nvidia_docs/nvidia_docs/spiders/output.json --output_path result/cleaned_data.txt
