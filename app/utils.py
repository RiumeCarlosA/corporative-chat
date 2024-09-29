import os
from PyPDF2 import PdfReader
from config.logging_config import setup_logging

logger = setup_logging()

def extract_text_from_pdf(file_path):
    """Extrai o texto de um arquivo PDF."""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            logger.info(f"Starting to process PDF: {file_path}")
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
            logger.info(f"Successfully extracted text from {file_path}")
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
    return text

def combine_documents():
    """Combine the text from all stored documents for context."""
    combined_text = ""
    folder = "models/"
    try:
        if not os.path.exists(folder):
            logger.error(f"Folder {folder} does not exist.")
            return None

        logger.info(f"Combining documents from folder: {folder}")
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename), 'r', encoding='utf-8') as file:
                    combined_text += file.read() + " "
        
        if combined_text:
            logger.info("Successfully combined documents.")
        else:
            logger.warning("No documents found to combine.")
    
    except Exception as e:
        logger.error(f"Error combining documents from {folder}: {e}")
    
    return combined_text if combined_text else None

