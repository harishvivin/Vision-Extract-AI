"""Regression tests for the generic medical document QA pipeline."""

import sys
from pathlib import Path
import tempfile
import unittest

import fitz

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.medical_question_parser import DocumentQuestionParser
from src.qa_engine import DocumentQAEngine

NOT_FOUND_MESSAGE = "The uploaded report does not contain this information."


class MedicalDocumentQATests(unittest.TestCase):
    def _create_sample_pdf(self, output_path: Path, text_blocks: list[str]) -> Path:
        doc = fitz.open()
        page = doc.new_page()
        for idx, block in enumerate(text_blocks, start=1):
            page.insert_text((40, 40 + idx * 30), block, fontsize=11)
        doc.save(output_path)
        doc.close()
        return output_path

    def test_question_parser_extracts_generic_keywords(self):
        parser = DocumentQuestionParser()

        self.assertEqual(parser.parse("What is the patient's name?").keywords, ["patient", "name"])
        self.assertEqual(parser.parse("What is the creatinine value?").keywords, ["creatinine"])
        self.assertEqual(parser.parse("What is the HbA1c?").keywords, ["hba1c"])
        self.assertEqual(parser.parse("Summarize this report").intent, "summary")

    def test_generic_qa_engine_answers_across_two_medical_pdfs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pdf_a = self._create_sample_pdf(
                tmp_path / "hospital_a.pdf",
                [
                    "Patient Name: John Doe",
                    "Creatinine: 1.2 mg/dL",
                    "Hemoglobin: 13.4 g/dL",
                    "Diagnosis: Mild anemia",
                ],
            )
            pdf_b = self._create_sample_pdf(
                tmp_path / "hospital_b.pdf",
                [
                    "Name: Jane Smith",
                    "HbA1c: 6.8%",
                    "Blood Pressure: 120/80",
                    "Diagnosis: Healthy",
                ],
            )

            engine = DocumentQAEngine(outputs_dir=tmp_path / "qa_out")
            for pdf_path in [pdf_a, pdf_b]:
                engine.purge_and_create_session(pdf_path, pdf_path.name)
                result_name = engine.ask("What is the patient's name?")
                self.assertNotEqual(result_name.answer, NOT_FOUND_MESSAGE)
                self.assertIn("Doe", result_name.answer) if pdf_path.name == "hospital_a.pdf" else self.assertIn("Smith", result_name.answer)

                result_creatinine = engine.ask("What is the creatinine value?")
                if pdf_path.name == "hospital_a.pdf":
                    self.assertIn("1.2", result_creatinine.answer)
                else:
                    self.assertEqual(result_creatinine.answer, NOT_FOUND_MESSAGE)

                result_hba1c = engine.ask("What is the HbA1c?")
                if pdf_path.name == "hospital_b.pdf":
                    self.assertIn("6.8", result_hba1c.answer)
                else:
                    self.assertEqual(result_hba1c.answer, NOT_FOUND_MESSAGE)

                summary = engine.ask("Summarize this report")
                self.assertNotEqual(summary.answer, NOT_FOUND_MESSAGE)


if __name__ == "__main__":
    unittest.main()
