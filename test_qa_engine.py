"""
Comprehensive Unit & Integration Test Suite for Medical PDF Document QA Pipeline.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import fitz  # PyMuPDF

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.question_parser import QuestionParser
from src.qa_engine import DocumentQAEngine, NOT_FOUND_ANSWER


class ComprehensiveDocumentQATests(unittest.TestCase):

    def _create_medical_pdf(self, file_path: Path, text_lines: list[str]) -> Path:
        """Helper to generate synthetic PDF document with given text lines."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y_offset = 50
        for line in text_lines:
            page.insert_text((50, y_offset), line, fontsize=12)
            y_offset += 25
        doc.save(file_path)
        doc.close()
        return file_path

    def test_question_parser_extracts_keywords_and_intent(self):
        parser = QuestionParser()

        parsed1 = parser.parse("What is the patient's name?")
        self.assertEqual(parsed1.intent, "lookup")
        self.assertIn("patient", parsed1.keywords)
        self.assertIn("name", parsed1.keywords)

        parsed2 = parser.parse("Summarize this medical report.")
        self.assertEqual(parsed2.intent, "summary")
        self.assertTrue(parsed2.is_summary_request)

        parsed3 = parser.parse("What is the creatinine value?")
        self.assertEqual(parsed3.intent, "lookup")
        self.assertIn("creatinine", parsed3.keywords)

    def test_qa_engine_across_multiple_hospital_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Hospital Format A (City Care Hospital)
            pdf_a = self._create_medical_pdf(
                tmp_path / "city_care_report.pdf",
                [
                    "City Care Hospital - Laboratory Report",
                    "Patient Name: Manjit Singh",
                    "Age: 45 | Sex: Male",
                    "Hemoglobin: 14.2 g/dL",
                    "Creatinine: 1.1 mg/dL",
                    "HbA1c: 5.7 %",
                    "Blood Pressure: 120/80 mmHg",
                    "Diagnosis: Normal baseline health evaluation",
                ]
            )

            # Hospital Format B (Apex Medical Center)
            pdf_b = self._create_medical_pdf(
                tmp_path / "apex_center_report.pdf",
                [
                    "Apex Diagnostic & Research Center",
                    "Name of Patient: Sarah Connor",
                    "Attending Hospital: Apex Medical Center",
                    "Hemoglobin Level: 11.5 g/dL",
                    "Serum Creatinine: 0.9 mg/dL",
                    "Glycated Hemoglobin (HbA1c): 6.4 %",
                    "BP Reading: 135/88",
                    "Final Diagnosis: Pre-diabetes and mild anemia",
                ]
            )

            engine = DocumentQAEngine(outputs_dir=tmp_path / "qa_outputs")

            # --- TEST PDF A ---
            engine.purge_and_create_session(pdf_a, pdf_a.name)

            res_name = engine.ask("What is the patient's name?")
            self.assertIn("Manjit", res_name.answer)

            res_hospital = engine.ask("What is the hospital?")
            self.assertIn("City Care", res_hospital.answer)

            res_hb = engine.ask("What is the hemoglobin level?")
            self.assertIn("14.2", res_hb.answer)

            res_creat = engine.ask("What is the creatinine value?")
            self.assertIn("1.1", res_creat.answer)

            res_hba1c = engine.ask("What is the HbA1c?")
            self.assertIn("5.7", res_hba1c.answer)

            res_bp = engine.ask("What is the blood pressure?")
            self.assertIn("120/80", res_bp.answer)

            res_diag = engine.ask("What is the diagnosis?")
            self.assertIn("Normal", res_diag.answer)

            res_missing = engine.ask("What is the HIV status?")
            self.assertEqual(res_missing.answer, NOT_FOUND_ANSWER)
            self.assertEqual(res_missing.confidence, 0.0)

            res_summary = engine.ask("Summarize this report.")
            self.assertNotEqual(res_summary.answer, NOT_FOUND_ANSWER)
            self.assertIn("City Care", res_summary.answer)

            # --- TEST PDF B (SESSION PURGING & ISOLATION) ---
            engine.purge_and_create_session(pdf_b, pdf_b.name)

            res_b_name = engine.ask("What is the patient's name?")
            self.assertIn("Sarah", res_b_name.answer)
            self.assertNotIn("Manjit", res_b_name.answer)  # Ensure zero data leak from PDF A!

            res_b_hosp = engine.ask("What is the hospital?")
            self.assertIn("Apex", res_b_hosp.answer)

            res_b_hb = engine.ask("What is the hemoglobin?")
            self.assertIn("11.5", res_b_hb.answer)

            res_b_hba1c = engine.ask("What is the HbA1c?")
            self.assertIn("6.4", res_b_hba1c.answer)

            res_b_diag = engine.ask("What is the diagnosis?")
            self.assertIn("Pre-diabetes", res_b_diag.answer)


if __name__ == "__main__":
    unittest.main()
