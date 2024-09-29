import os
from app.utils import extract_text_from_pdf
from app.models import save_document_text
from tasks.pdf_processing import process_pdf_async
from config.logging_config import setup_logging

def upload_files(files):
    """Carrega arquivos PDFs e inicia processamento assíncrono"""
    if not files:
        logger.warning("No files provided for upload.")
        return

    for file in files:
        file_path = os.path.join('uploads', file.filename)
        try:
            file.save(file_path)
            logger.info(f"File {file.filename} saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}")
            continue

        try:
            process_pdf_async.delay(file_path)
            logger.info(f"Asynchronous processing initiated for {file.filename}")
        except Exception as e:
            logger.error(f"Failed to start asynchronous processing for {file.filename}: {e}")
