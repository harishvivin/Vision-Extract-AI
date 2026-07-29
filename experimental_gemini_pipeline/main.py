"""
Main Orchestration Script for Experimental Gemini PDF Localization Pipeline.
Processes medical report PDFs across 10 target queries:
Patient Name, Hospital Name, Creatinine, HbA1c, Hemoglobin, Blood Pressure, Diagnosis, ECG, HIV, Summary.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from .config import OUTPUTS_DIR, PROJECT_ROOT
from .gemini_client import ExperimentalGeminiClient
from .coordinate_cropper import crop_pdf_region

logger = logging.getLogger("experimental_main")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Standard test queries requested
DEFAULT_QUESTIONS = [
    "Patient Name",
    "Hospital Name",
    "Creatinine",
    "HbA1c",
    "Hemoglobin",
    "Blood Pressure",
    "Diagnosis",
    "ECG",
    "HIV",
    "Summary"
]


class ExperimentalPipeline:
    """Master Pipeline orchestrating Gemini PDF localization & PyMuPDF crop extraction."""

    def __init__(self, output_dir: Path = OUTPUTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = ExperimentalGeminiClient()

    def process_document(self, pdf_path: str | Path, questions: List[str] = DEFAULT_QUESTIONS) -> Dict[str, Any]:
        """
        Process a single PDF document across a list of target queries.

        Args:
            pdf_path (str | Path): Input PDF file path.
            questions (List[str]): List of natural language questions or requested fields.

        Returns:
            Dict[str, Any]: Comprehensive localization report mapping each question to page, box, text, & crop PNG.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Input PDF document does not exist: {pdf_path}")

        logger.info(f"=== Starting Experimental Gemini Pipeline for '{pdf_path.name}' ===")
        doc_output_dir = self.output_dir / pdf_path.stem
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for idx, q in enumerate(questions, 1):
            logger.info(f"[{idx}/{len(questions)}] Querying Gemini for: '{q}'...")
            loc_data = self.client.query_pdf(pdf_path, q)

            found = loc_data.get("found", False)
            page_num = loc_data.get("page", 1)
            bbox = loc_data.get("bounding_box", {})
            matched_text = loc_data.get("matched_text", "")
            confidence = loc_data.get("confidence", 0.0)

            crop_filename = None
            crop_path = None

            if found and bbox:
                safe_q_name = "".join([c if c.isalnum() else "_" for c in q]).lower()
                crop_filename = f"crop_{idx:02d}_{safe_q_name}.png"
                target_crop_file = doc_output_dir / crop_filename

                try:
                    crop_pdf_region(
                        pdf_path=pdf_path,
                        page_number=page_num,
                        bounding_box=bbox,
                        output_path=target_crop_file
                    )
                    crop_path = str(target_crop_file)
                except Exception as e_crop:
                    logger.error(f"Failed to render crop for '{q}': {e_crop}")

            item_result = {
                "question": q,
                "found": found,
                "page": page_num,
                "bounding_box": bbox,
                "matched_text": matched_text,
                "confidence": confidence,
                "crop_filename": crop_filename,
                "crop_path": crop_path
            }
            results.append(item_result)

        summary_report = {
            "document_name": pdf_path.name,
            "total_queries": len(questions),
            "found_count": sum(1 for r in results if r["found"]),
            "results": results
        }

        report_file = doc_output_dir / "pipeline_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=2)

        logger.info(f"=== Completed Experimental Pipeline for '{pdf_path.name}'. Report saved to '{report_file}' ===")
        return summary_report


def main():
    """CLI Entrypoint for running the experimental Gemini pipeline."""
    default_pdf = PROJECT_ROOT / "INPUT_images_and_questions.pdf"

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = default_pdf

    if not pdf_path.exists():
        print(f"Usage: python -m experimental_gemini_pipeline.main <path_to_pdf>")
        print(f"Error: Sample PDF not found at {pdf_path}")
        sys.exit(1)

    pipeline = ExperimentalPipeline()
    report = pipeline.process_document(pdf_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
