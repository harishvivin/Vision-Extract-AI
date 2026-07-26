"""Generic medical document processing pipeline.

This pipeline indexes PDF text blocks for retrieval QA instead of relying on
object-detection prompts or hardcoded medical field extraction.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from config import LOGS_DIR, OUTPUTS_DIR
from src.medical_question_parser import DocumentQuestionParser
from src.pdf_reader import PDFReader, PageData
from src.utils import create_output_zip, log_detection_data

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)


@dataclass
class ProcessedPageResult:
    """Result data structure for a single indexed PDF page."""

    page_number: int
    raw_question: str
    parsed_question: Dict[str, Any]
    detection_prompt: str
    confidence: float
    bounding_box: List[float]
    spatial_score: float
    sam2_used: bool
    processing_time_ms: float
    output_filename: str
    output_image_path: str
    overlay_image: Image.Image
    cropped_image: Image.Image


class ExtractionPipeline:
    """Index a PDF by page text and create generic page screenshots."""

    def __init__(self, output_dir: Path = OUTPUTS_DIR, log_dir: Path = LOGS_DIR):
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.pdf_reader = PDFReader()
        self.question_parser = DocumentQuestionParser()

    def run(self, pdf_path: str | Path, progress_callback=None) -> List[ProcessedPageResult]:
        start_total_time = time.time()
        pdf_path = Path(pdf_path)
        logger.info("=== Starting generic document pipeline for: %s ===", pdf_path)

        if progress_callback:
            progress_callback(10, "Extracting pages and text from PDF...")
        pages_data = self.pdf_reader.extract_all(pdf_path)

        total_pages = len(pages_data)
        results: List[ProcessedPageResult] = []

        for idx, page_data in enumerate(pages_data):
            page_num = page_data.page_number
            msg = f"Processing page {page_num} of {total_pages}"
            logger.info("--- %s ---", msg)
            if progress_callback:
                prog_pct = 15 + int((idx / total_pages) * 75) if total_pages else 100
                progress_callback(prog_pct, msg)

            page_start_time = time.time()
            parsed_q = self.question_parser.parse(page_data.question_text or page_data.raw_text)

            image = page_data.page_image
            output_filename = f"page_{page_num}.png"
            output_path = self.output_dir / output_filename
            image.save(output_path, format="PNG")

            previews_dir = self.output_dir / "previews"
            previews_dir.mkdir(parents=True, exist_ok=True)
            preview_path = previews_dir / f"preview_page_{page_num}.png"
            image.save(preview_path, format="PNG")

            total_page_time_ms = (time.time() - page_start_time) * 1000
            log_data = {
                "page_number": page_num,
                "raw_question": page_data.question_text,
                "parsed_question": {
                    "object": None,
                    "color": None,
                    "position": None,
                    "filename": output_filename,
                    "keywords": parsed_q.keywords,
                    "intent": parsed_q.intent,
                },
                "detection_prompt": "text-block-indexing",
                "confidence": 1.0,
                "bounding_box": [0.0, 0.0, 1.0, 1.0],
                "spatial_score": 1.0,
                "sam2_used": False,
                "processing_time_ms": total_page_time_ms,
                "output_filename": output_filename,
                "output_path": str(output_path),
                "attempts_log": [],
            }
            log_detection_data(log_data, log_file=self.log_dir / "detections.json")

            result = ProcessedPageResult(
                page_number=page_num,
                raw_question=page_data.question_text,
                parsed_question=log_data["parsed_question"],
                detection_prompt="text-block-indexing",
                confidence=1.0,
                bounding_box=[0.0, 0.0, 1.0, 1.0],
                spatial_score=1.0,
                sam2_used=False,
                processing_time_ms=total_page_time_ms,
                output_filename=output_filename,
                output_image_path=str(output_path),
                overlay_image=image.copy(),
                cropped_image=image.copy(),
            )
            results.append(result)

        if progress_callback:
            progress_callback(95, "Packaging extracted images into ZIP archive...")
        zip_path = create_output_zip(self.output_dir)
        if not zip_path.exists():
            logger.warning("ZIP archive creation failed; continuing without zip file.")

        total_elapsed = time.time() - start_total_time
        logger.info("=== Pipeline completed successfully in %.2f seconds. Output zip: %s ===", total_elapsed, zip_path)

        if progress_callback:
            progress_callback(100, "Processing Complete!")

        return results
