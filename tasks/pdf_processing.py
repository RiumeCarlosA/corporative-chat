from celery import shared_task
from app.utils import extract_text_from_pdf
from app.models import save_document_text, BERTModel, treat_text_dynamically
from elasticsearch import Elasticsearch
import os
from config.logging_config import setup_logging

logger = setup_logging()
es = Elasticsearch([{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}])

@shared_task
def process_pdf_async(file_path):
    """Processa um PDF de forma assíncrona, trata o texto dinamicamente e indexa no Elasticsearch."""
    try:
        logger.info(f"Processing PDF: {file_path}")
        text = extract_text_from_pdf(file_path)
        if text:
            filename = os.path.basename(file_path).replace('.pdf', '')
            treated_text = treat_text_dynamically(text)

            # Indexar no Elasticsearch
            es.index(index='company_docs', id=filename, body=treated_text)

            # Salvar o texto processado (se necessário localmente)
            save_document_text(str(treated_text), filename)

            logger.info(f"Fine-tuning initiated with text from: {file_path}")
            return treated_text
        else:
            logger.warning(f"No text extracted from {file_path}")
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}")