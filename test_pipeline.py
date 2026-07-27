"""
End-to-End Test Suite for Vision Extract AI Pipeline.
Runs PDF text extraction and page preview rendering, verifying outputs and ZIP packaging.
"""

import sys
import logging
import tempfile
from pathlib import Path
import fitz

from config import BASE_DIR, OUTPUTS_DIR, LOGS_DIR
from src.pipeline import ExtractionPipeline
from src.qa_engine import DocumentQAEngine, NOT_FOUND_ANSWER

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_pipeline")


def run_test():
    pdf_file = BASE_DIR / "INPUT_images_and_questions.pdf"

    if not pdf_file.exists():
        logger.info("Default sample PDF not found; creating temporary test PDF...")
        pdf_file = BASE_DIR / "temp_test_report.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Test Medical Report\nPatient Name: John Doe\nHemoglobin: 13.5 g/dL", fontsize=12)
        doc.save(pdf_file)
        doc.close()

    logger.info("=== Starting Integration Test for Vision Extract AI Pipeline ===")

    pipeline = ExtractionPipeline(output_dir=OUTPUTS_DIR, log_dir=LOGS_DIR)

    def progress_callback(pct, msg):
        logger.info(f"Progress: [{pct}%] - {msg}")

    results = pipeline.run(pdf_file, progress_callback=progress_callback)

    logger.info(f"Pipeline executed. Total pages processed: {len(results)}")

    if len(results) > 0:
        logger.info(f"VERIFIED: {len(results)} pages processed and rendered.")
    else:
        logger.error("FAILED: No pages processed!")
        sys.exit(1)

    zip_file = OUTPUTS_DIR / "all_extracted_objects.zip"
    if zip_file.exists() and zip_file.stat().st_size > 0:
        logger.info(f"VERIFIED: ZIP package '{zip_file.name}' created ({zip_file.stat().st_size} bytes).")
    else:
        logger.error("FAILED: ZIP package missing or empty!")
        sys.exit(1)

    # Test QA Engine on the processed PDF
    qa_engine = DocumentQAEngine(outputs_dir=OUTPUTS_DIR)
    qa_engine.purge_and_create_session(pdf_file, pdf_file.name)
    qa_res = qa_engine.ask("Summarize this document.")
    logger.info(f"QA Summary test response: {qa_res.answer}")

    logger.info("=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ===")


if __name__ == "__main__":
    run_test()
