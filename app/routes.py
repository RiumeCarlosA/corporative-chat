from flask import Blueprint, request, jsonify
from tasks.pdf_processing import process_pdf_async
from app.models import BERTModel
import os
from config.logging_config import setup_logging

api = Blueprint('api', __name__)

bert_model = BERTModel()

logger = setup_logging()

@api.route("/upload_pdfs", methods=["POST"])
def upload_pdfs():
    files = request.files.getlist("file[]")
    if not files:
        logger.warning("No files uploaded.")
        return jsonify({"error": "No files uploaded"}), 400

    for file in files:
        file_path = os.path.join("uploads", file.filename)
        try:
            file.save(file_path)
            logger.info(f"File {file.filename} saved successfully at {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
            return jsonify({"error": f"Failed to save {file.filename}"}), 500

        # Process PDF asynchronously
        try:
            process_pdf_async.delay(file_path)
            logger.info(f"File {file.filename} sent for asynchronous processing.")
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            return jsonify({"error": f"Failed to process {file.filename}"}), 500

    return jsonify({"message": "PDFs uploaded and sent for processing."})

@api.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get('question')

    if not question:
        logger.warning("No question provided in the request.")
        return jsonify({"error": "No question provided"}), 400

    try:
        logger.info(f"Received question: {question}")
        answer = bert_model.ask_question(question)
        logger.info(f"Answer generated: {answer}")
        return jsonify({"answer": answer})
    except ValueError as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({"error": str(e)}), 400

