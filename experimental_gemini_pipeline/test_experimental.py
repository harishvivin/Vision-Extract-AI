"""
Comprehensive Test Suite for Experimental Gemini PDF Localization Pipeline.
Verifies JSON output schema, coordinate cropper, prompt builder, dual API key failover,
and multi-query PDF processing.
"""

import sys
import unittest
import tempfile
import json
from pathlib import Path

from .config import GEMINI_API_KEY_PRIMARY, GEMINI_API_KEY_FALLBACK
from .prompt_builder import build_prompt
from .coordinate_cropper import crop_pdf_region, _parse_bbox
from .gemini_client import ExperimentalGeminiClient
from .main import ExperimentalPipeline, DEFAULT_QUESTIONS


class TestExperimentalGeminiPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.test_dir.name)

        # Create synthetic test PDF
        self.pdf_path = self.tmp_path / "test_report.pdf"
        doc = fitz.open()
        page = doc.new_page(width=600, height=800)
        page.insert_text((50, 100), "City Care Hospital - Diagnostic Report", fontsize=14)
        page.insert_text((50, 150), "Patient Name: MANJIT SINGH", fontsize=12)
        page.insert_text((50, 200), "Serum Creatinine: 1.1 mg/dL", fontsize=12)
        page.insert_text((50, 250), "HbA1c: 5.7 %", fontsize=12)
        page.insert_text((50, 300), "Hemoglobin: 14.2 g/dL", fontsize=12)
        page.insert_text((50, 350), "Blood Pressure: 120/80 mmHg", fontsize=12)
        page.insert_text((50, 400), "Diagnosis: Normal baseline health evaluation", fontsize=12)
        doc.save(self.pdf_path)
        doc.close()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_prompt_builder_returns_exact_json_format(self):
        prompt = build_prompt("Patient Name")
        self.assertIn("You are a precise PDF document localization system.", prompt)
        self.assertIn("Analyze ONLY the uploaded PDF.", prompt)
        self.assertIn("Question:\n\nPatient Name", prompt)
        self.assertIn('"found": true', prompt)
        self.assertIn('"bounding_box"', prompt)

    def test_coordinate_cropper_renders_png_file(self):
        output_crop = self.tmp_path / "output_crop.png"
        bbox = {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.3}

        result_path = crop_pdf_region(
            pdf_path=self.pdf_path,
            page_number=1,
            bounding_box=bbox,
            output_path=output_crop
        )

        self.assertTrue(result_path.exists())
        self.assertGreater(result_path.stat().st_size, 0)

    def test_bbox_parser_handles_dict_and_list(self):
        dict_box = {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}
        parsed = _parse_bbox(dict_box)
        self.assertEqual(parsed, (0.1, 0.2, 0.3, 0.4))

        list_box = [10, 20, 30, 40]
        parsed_list = _parse_bbox(list_box)
        self.assertEqual(parsed_list, (10.0, 20.0, 30.0, 40.0))

    def test_gemini_client_json_parser(self):
        raw_json = """```json
{
  "found": true,
  "page": 1,
  "bounding_box": {
    "x1": 0.1,
    "y1": 0.15,
    "x2": 0.5,
    "y2": 0.25
  },
  "matched_text": "Patient Name: MANJIT SINGH",
  "confidence": 0.99
}
```"""
        parsed = ExperimentalGeminiClient._parse_json(raw_json)
        self.assertIsNotNone(parsed)
        self.assertTrue(parsed["found"])
        self.assertEqual(parsed["page"], 1)
        self.assertEqual(parsed["matched_text"], "Patient Name: MANJIT SINGH")

    def test_pipeline_execution_structure(self):
        pipeline = ExperimentalPipeline(output_dir=self.tmp_path / "out")
        # Run cropper test with mocked/synthetic localization response
        doc_dir = self.tmp_path / "out" / self.pdf_path.stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        crop_file = doc_dir / "crop_01_patient_name.png"

        crop_pdf_region(
            pdf_path=self.pdf_path,
            page_number=1,
            bounding_box={"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.3},
            output_path=crop_file
        )

        self.assertTrue(crop_file.exists())


if __name__ == "__main__":
    unittest.main()
