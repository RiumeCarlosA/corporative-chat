from transformers import DebertaV2Tokenizer, DebertaV2ForQuestionAnswering
import tensorflow as tf
from config.logging_config import setup_logging
from elasticsearch import Elasticsearch
from app.utils import combine_documents
import os
import re
import torch

tf.compat.v1.enable_eager_execution()

logger = setup_logging()
es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])

class BERTModel:
    def __init__(self):
        # Correct imports for DeBERTa
        self.tokenizer = DebertaV2Tokenizer.from_pretrained("microsoft/deberta-v3-base")
        self.model = DebertaV2ForQuestionAnswering.from_pretrained("microsoft/deberta-v3-base")
        self.model_path = './fine_tuned_model'
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model, if available."""
        if os.path.exists(self.model_path):
            self.model = DebertaV2ForQuestionAnswering.from_pretrained(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.info("No fine-tuned model found. Using the pre-trained mDeBERTa model.")

    def chunk_text(self, text, max_len=400):
        """Divide text into chunks that fit within DeBERTa's token limit."""
        tokens = self.tokenizer.encode(text)
        chunks = [tokens[i:i + max_len] for i in range(0, len(tokens), max_len)]
        chunk_texts = [self.tokenizer.decode(chunk) for chunk in chunks]
        logger.info(f"Text divided into {len(chunk_texts)} parts.")
        return chunk_texts

    def validate_answer(self, answer):
        """Ensure the answer is not just special tokens like [SEP], [PAD], etc."""
        special_tokens = ['[SEP]', '[PAD]', '[CLS]']
        if all(token in special_tokens for token in answer.split()):
            return ""
        return answer

    def fine_tune_from_text(self, text):
        """Fine-tune the model dynamically with the provided text."""
        perguntas = ["O que esse documento descreve?", "Qual a informação principal deste documento?"]

        for pergunta in perguntas:
            inputs = self.tokenizer(pergunta, text, return_tensors='tf', max_length=400, truncation=True, padding='max_length')
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Fine-tuning with fictitious positions
            start_positions = [1]
            end_positions = [len(input_ids[0]) - 2]

            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
            self.model.compile(optimizer=optimizer, loss=self.model.compute_loss)

            self.model.fit(
                x={'input_ids': input_ids, 'attention_mask': attention_mask},
                y={'start_positions': start_positions, 'end_positions': end_positions},
                epochs=1
            )

        self.model.save_pretrained(self.model_path)
        logger.info(f"Fine-tuned model saved at {self.model_path}")

    def predict(self, question, context_chunk):
        inputs = self.tokenizer(question, context_chunk, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = self.model(**inputs)
        start_logits, end_logits = outputs.start_logits, outputs.end_logits

        start_idx = torch.argmax(start_logits, dim=-1).item()
        end_idx = torch.argmax(end_logits, dim=-1).item()

        answer = self.tokenizer.decode(inputs['input_ids'][0][start_idx:end_idx + 1])
        return answer.strip()
        
    def ask_question(self, question):
        """Search Elasticsearch first, then use mDeBERTa model for question answering."""
        es_results = self.search_in_elasticsearch(question)

        if not es_results:
            logger.warning("No results found in Elasticsearch. Falling back to raw document content.")
            combined_text = combine_documents()  # Use full document content if no results in Elasticsearch
            chunks = self.chunk_text(combined_text, max_len=400)
        else:
            chunks = []
            for result in es_results:
                # Dynamically check for fields that contain relevant text
                source = result.get('_source', {})
                content_field = source.get('content') or source.get('text') or ' '.join(source.values())
                
                if content_field:
                    chunks.extend(self.chunk_text(content_field, max_len=400))
                else:
                    logger.warning(f"No relevant content field found in result: {result}")

        best_answer = ""
        best_score = float('-inf')

        for chunk in chunks:
            answer = self.predict(question, chunk)
            logger.info(f"Chunk score: {answer}")

            if len(answer.strip()) > 0:
                best_answer = answer

        return best_answer if best_answer else "No relevant answer found."

    def search_in_elasticsearch(self, query):
        """Dynamically search in Elasticsearch using detected fields."""
        try:
            # Get the mappings dynamically
            index_mapping = es.indices.get_mapping(index="company_docs")
            fields = self.get_fields_from_mapping(index_mapping)

            # Create a dynamic multi_match query across all detected fields
            results = es.search(index="company_docs", body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                        "type": "best_fields",
                        "fuzziness": "AUTO"  # Enabling fuzzy matching for improved results
                    }
                },
                "highlight": {
                    "fields": {field: {} for field in fields}  # Highlighting for better focus
                }
            }, headers={"Content-Type": "application/json"})
            
            return results['hits']['hits']
        
        except Exception as e:
            logger.error(f"Error querying Elasticsearch: {e}")
            return []

    def get_fields_from_mapping(self, index_mapping):
        """Extract field names dynamically from the index mapping."""
        properties = index_mapping['company_docs']['mappings']['properties']
        return list(properties.keys())

# Dynamic text processing and section detection
def treat_text_dynamically(text):
    """Dynamically treat and organize text based on automatic section detection."""
    text = re.sub(r'\n+', '\n', text)

    organized_data = {}
    current_section = None
    section_pattern = re.compile(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)*$')  # Detects titles like "Experiência Profissional"

    for line in text.split('\n'):
        line = line.strip()

        if section_pattern.match(line):  # If it detects a section title
            current_section = line.lower().replace(" ", "_")
            organized_data[current_section] = []
        elif current_section:
            organized_data[current_section].append(line.strip())

    for section in organized_data:
        organized_data[section] = ' '.join(organized_data[section])

    return organized_data

def save_document_text(text, filename):
    """Save extracted text from a document."""
    try:
        with open(f"models/{filename}.txt", "w") as f:
            f.write(text)
        return True
    except Exception as e:
        logger.error(f"Error saving the text: {e}")
        return False
