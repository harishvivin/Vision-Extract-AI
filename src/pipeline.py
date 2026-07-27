"""
Document Processing & Inspection Pipeline.
Processes input PDF documents, renders high-resolution page previews,
and packages extracted page outputs.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from PIL import Image

from config import LOGS_DIR, OUTPUTS_DIR
from src.pdf_reader import PDFReader
from src.question_parser import QuestionParser
from src.utils import create_output_zip, log_detection_data

logger = logging.getLogger("pipeline")


@dataclass
class ProcessedPageResult:
    """Result data structure for a single processed PDF page."""
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
    """Processes PDF documents to extract page renderings and logs indexing status."""

    def __init__(self, output_dir: Path = OUTPUTS_DIR, log_dir: Path = LOGS_DIR):
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.previews_dir = self.output_dir / "previews"
        self.previews_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_reader = PDFReader()
        self.question_parser = QuestionParser()

    def run(self, pdf_path: str | Path, progress_callback=None) -> List[ProcessedPageResult]:
        """
        Execute document extraction and preview rendering.

        Args:
            pdf_path (str | Path): Input PDF path.
            progress_callback (Optional[Callable]): Callback for UI progress updates.

        Returns:
            List[ProcessedPageResult]: Processed result per PDF page.
        """
        start_total_time = time.time()
        pdf_path = Path(pdf_path)
        logger.info(f"=== Running Document Extraction Pipeline for: {pdf_path} ===")

        if progress_callback:
            progress_callback(10, "Extracting page renderings from PDF...")

        pages_data = self.pdf_reader.extract_all(pdf_path)
        total_pages = len(pages_data)
        results: List[ProcessedPageResult] = []

        for idx, page_data in enumerate(pages_data):
            page_num = page_data.page_number
            msg = f"Processing & indexing page {page_num} of {total_pages}"
            logger.info(f"--- {msg} ---")

            if progress_callback:
                prog_pct = 15 + int((idx / max(1, total_pages)) * 75)
                progress_callback(prog_pct, msg)

            page_start_time = time.time()
            parsed_q = self.question_parser.parse(page_data.question_text or page_data.raw_text)

            image = page_data.page_image
            output_filename = f"page_{page_num}.png"
            output_path = self.output_dir / output_filename
            image.save(output_path, format="PNG", compress_level=1)

            preview_filename = f"preview_page_{page_num}.png"
            preview_path = self.previews_dir / preview_filename
            image.save(preview_path, format="PNG", compress_level=1)

            proc_time_ms = (time.time() - page_start_time) * 1000

            parsed_dict = {
                "object": None,
                "color": None,
                "position": None,
                "filename": output_filename,
                "keywords": parsed_q.keywords,
                "intent": parsed_q.intent,
            }

            log_data = {
                "page_number": page_num,
                "raw_question": page_data.question_text,
                "parsed_question": parsed_dict,
                "detection_prompt": "text-block-indexing",
                "confidence": 1.0,
                "bounding_box": [0.0, 0.0, 1.0, 1.0],
                "spatial_score": 1.0,
                "sam2_used": False,
                "processing_time_ms": proc_time_ms,
                "output_filename": output_filename,
                "output_path": str(output_path),
                "attempts_log": [],
            }
            log_detection_data(log_data, log_file=self.log_dir / "detections.json")

            result = ProcessedPageResult(
                page_number=page_num,
                raw_question=page_data.question_text,
                parsed_question=parsed_dict,
                detection_prompt="text-block-indexing",
                confidence=1.0,
                bounding_box=[0.0, 0.0, 1.0, 1.0],
                spatial_score=1.0,
                sam2_used=False,
                processing_time_ms=proc_time_ms,
                output_filename=output_filename,
                output_image_path=str(output_path),
                overlay_image=image.copy(),
                cropped_image=image.copy()
            )
            results.append(result)

        if progress_callback:
            progress_callback(95, "Completed document processing...")

        total_elapsed = time.time() - start_total_time
        logger.info(f"=== Document Pipeline completed in {total_elapsed:.2f}s ===")

        if progress_callback:
            progress_callback(100, "Document Indexing & Analysis Complete!")

        return results
