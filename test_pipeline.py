"""
End-to-End Test Suite for Vision Extract AI Pipeline.
Runs full extraction on INPUT_images_and_questions.pdf and validates output files.
"""

import sys
import logging
from pathlib import Path

from config import BASE_DIR, OUTPUTS_DIR, LOGS_DIR
from src.pipeline import ExtractionPipeline

# Setup Logging to stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("test_pipeline")


def run_test():
    pdf_file = BASE_DIR / "INPUT_images_and_questions.pdf"
    if not pdf_file.exists():
        logger.error(f"Input PDF not found at {pdf_file}")
        sys.exit(1)

    logger.info("=== Starting Integration Test for Vision Extract AI Pipeline ===")

    pipeline = ExtractionPipeline(output_dir=OUTPUTS_DIR, log_dir=LOGS_DIR)

    def progress_callback(pct, msg):
        logger.info(f"Progress: [{pct}%] - {msg}")

    results = pipeline.run(pdf_file, progress_callback=progress_callback)

    logger.info(f"Pipeline executed. Total pages processed: {len(results)}")

    expected_files = [
        "01_yellow_tulips.png",
        "02_red_tulips.png",
        "03_silver_car.png",
        "04_green_balloon.png",
        "05_pears.png",
        "06_middle_braid.png",
        "07_front_sheep.png",
        "08_green_duck.png",
        "09_green_macaron.png",
        "10_flamingo.png"
    ]

    all_passed = True
    for expected in expected_files:
        out_path = OUTPUTS_DIR / expected
        if out_path.exists() and out_path.stat().st_size > 0:
            logger.info(f"VERIFIED: Output file '{expected}' created successfully ({out_path.stat().st_size} bytes).")
        else:
            logger.error(f"FAILED: Expected output file '{expected}' missing or empty!")
            all_passed = False

    zip_file = OUTPUTS_DIR / "all_extracted_objects.zip"
    if zip_file.exists():
        logger.info(f"VERIFIED: ZIP package '{zip_file.name}' created ({zip_file.stat().st_size} bytes).")
    else:
        logger.error("FAILED: ZIP package missing!")
        all_passed = False

    if all_passed:
        logger.info("=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ===")
    else:
        logger.error("=== SOME TESTS FAILED! ===")
        sys.exit(1)


if __name__ == "__main__":
    run_test()
