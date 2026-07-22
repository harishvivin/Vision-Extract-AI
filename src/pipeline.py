"""
Master Extraction Pipeline Orchestrator.
Integrates PDFReader, QuestionParser, GroundingDINODetector, SAM2Segmenter, and ImageCropper.
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image

from config import OUTPUTS_DIR, LOGS_DIR
from src.pdf_reader import PDFReader, PageData
from src.question_parser import QuestionParser, ParsedQuestion
from src.detector import GroundingDINODetector, DetectionResult
from src.segmenter import SAM2Segmenter
from src.cropper import ImageCropper
from src.utils import draw_detection_overlay, log_detection_data, create_output_zip

# Set up logger
logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)


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
    """Master AI Pipeline orchestrating PDF processing, NLP parsing, detection, SAM2 segmentation, and cropping."""

    def __init__(self, output_dir: Path = OUTPUTS_DIR, log_dir: Path = LOGS_DIR):
        """
        Initialize pipeline components.

        Args:
            output_dir (Path): Directory to save cropped output images.
            log_dir (Path): Directory for logging detection telemetry.
        """
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)

        # Initialize core modular sub-components
        self.pdf_reader = PDFReader()
        self.question_parser = QuestionParser()
        self.detector = GroundingDINODetector()
        self.segmenter = SAM2Segmenter()
        self.cropper = ImageCropper(output_dir=self.output_dir)

    def run(self, pdf_path: str | Path, progress_callback=None) -> List[ProcessedPageResult]:
        """
        Execute the end-to-end extraction pipeline on a target PDF file.

        Args:
            pdf_path (str | Path): Path to input PDF file.
            progress_callback (Optional[callable]): Callback function for tracking progress (0 to 100%).

        Returns:
            List[ProcessedPageResult]: Processing results for all document pages.
        """
        start_total_time = time.time()
        pdf_path = Path(pdf_path)
        logger.info(f"=== Starting Extraction Pipeline for: {pdf_path} ===")

        # Step 1: Extract PDF pages and questions
        if progress_callback:
            progress_callback(10, "Extracting pages and question text from PDF...")
        pages_data = self.pdf_reader.extract_all(pdf_path)

        total_pages = len(pages_data)
        results: List[ProcessedPageResult] = []

        for idx, page_data in enumerate(pages_data):
            page_num = page_data.page_number
            msg = f"Processing page {page_num} of {total_pages} ({page_data.question_text[:40]}...)"
            logger.info(f"--- {msg} ---")
            
            if progress_callback:
                prog_pct = 15 + int((idx / total_pages) * 75)
                progress_callback(prog_pct, msg)

            page_start_time = time.time()

            # Step 2: Parse question
            parsed_q = self.question_parser.parse(page_data.question_text)

            # Step 3: Zero-shot object detection with Grounding DINO
            det_res = self.detector.detect(page_data.extracted_photo, parsed_q)

            # Step 4: SAM2 segmentation or bounding box mask
            mask, sam_used = self.segmenter.generate_mask(page_data.extracted_photo, det_res.box)

            # Step 5: Crop and save output image
            cropped_img, saved_path = self.cropper.crop_and_save(
                image=page_data.extracted_photo,
                box=det_res.box,
                mask=mask,
                filename=parsed_q.filename
            )

            # Step 6: Create visualization overlay
            overlay_img = draw_detection_overlay(
                image=page_data.extracted_photo,
                box=det_res.box,
                label=parsed_q.primary_prompt,
                confidence=det_res.confidence,
                mask=mask if sam_used else None
            )

            # Save visual overlay preview image
            previews_dir = self.output_dir / "previews"
            previews_dir.mkdir(parents=True, exist_ok=True)
            overlay_img.save(previews_dir / f"preview_page_{page_num}.png", format="PNG")

            total_page_time_ms = (time.time() - page_start_time) * 1000

            # Step 7: Telemetry & Logging
            log_data = {
                "page_number": page_num,
                "raw_question": page_data.question_text,
                "parsed_question": {
                    "object": parsed_q.object,
                    "color": parsed_q.color,
                    "position": parsed_q.position,
                    "filename": parsed_q.filename
                },
                "detection_prompt": det_res.prompt_used,
                "confidence": det_res.confidence,
                "bounding_box": det_res.box,
                "spatial_score": det_res.spatial_score,
                "sam2_used": sam_used,
                "processing_time_ms": total_page_time_ms,
                "output_filename": parsed_q.filename,
                "output_path": str(saved_path),
                "attempts_log": det_res.attempts_log
            }
            log_detection_data(log_data, log_file=self.log_dir / "detections.json")

            result = ProcessedPageResult(
                page_number=page_num,
                raw_question=page_data.question_text,
                parsed_question={
                    "object": parsed_q.object,
                    "color": parsed_q.color,
                    "position": parsed_q.position,
                    "filename": parsed_q.filename
                },
                detection_prompt=det_res.prompt_used,
                confidence=det_res.confidence,
                bounding_box=det_res.box,
                spatial_score=det_res.spatial_score,
                sam2_used=sam_used,
                processing_time_ms=total_page_time_ms,
                output_filename=parsed_q.filename,
                output_image_path=str(saved_path),
                overlay_image=overlay_img,
                cropped_image=cropped_img
            )
            results.append(result)

        # Step 8: Create final ZIP package
        if progress_callback:
            progress_callback(95, "Packaging extracted images into ZIP archive...")
        zip_path = create_output_zip(self.output_dir)

        total_elapsed = time.time() - start_total_time
        logger.info(f"=== Pipeline completed successfully in {total_elapsed:.2f} seconds. Output zip: {zip_path} ===")

        if progress_callback:
            progress_callback(100, "Processing Complete!")

        return results
