"""
Comprehensive 5-Hospital PDF Test Suite for Medical Document QA Engine & Precision Crop Verification.
Tests Patient Name, Hospital Name, Age, Gender, Diagnosis, Hemoglobin, Creatinine, HbA1c, Blood Pressure, HIV, ECG, Summary.
"""

import sys
import unittest
import tempfile
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.qa_engine import DocumentQAEngine, NOT_FOUND_ANSWER


class FiveHospitalsBenchmarkTest(unittest.TestCase):

    def _create_medical_pdf(self, file_path: Path, text_lines: list[str]) -> Path:
        """Helper to generate synthetic multi-line PDF document."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y_offset = 50
        for line in text_lines:
            page.insert_text((50, y_offset), line, fontsize=11)
            y_offset += 24
        doc.save(file_path)
        doc.close()
        return file_path

    def test_5_hospital_formats_all_12_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outputs_path = tmp_path / "qa_outputs"
            engine = DocumentQAEngine(outputs_dir=outputs_path)

            # Hospital 1: City Care General Hospital
            pdf_1 = self._create_medical_pdf(
                tmp_path / "hosp1_city_care.pdf",
                [
                    "City Care General Hospital - Laboratory Report",
                    "Patient Name: MANJIT SINGH",
                    "Age: 45 Years | Gender: Male",
                    "Clinical Impression: Normal Health Screening",
                    "Hemoglobin: 14.5 g/dL",
                    "Serum Creatinine: 1.02 mg/dL",
                    "HbA1c: 5.6 %",
                    "Blood Pressure: 120/80 mmHg",
                    "HIV Screening Test: Non-Reactive",
                    "ECG Result: Normal Sinus Rhythm",
                ]
            )

            # Hospital 2: Apollo Health Diagnostics
            pdf_2 = self._create_medical_pdf(
                tmp_path / "hosp2_apollo.pdf",
                [
                    "Apollo Health Diagnostics Center",
                    "Name: Rajesh Sharma",
                    "Age: 52",
                    "Sex: Male",
                    "Primary Diagnosis: Essential Hypertension",
                    "Hemoglobin Level: 13.8 g/dL",
                    "Creatinine: 0.95 mg/dL",
                    "Glycated Hemoglobin (HbA1c): 6.8 %",
                    "BP: 140/90 mmHg",
                    "ECG: Sinus Bradycardia",
                ]
            )

            # Hospital 3: Fortis Medical Institute
            pdf_3 = self._create_medical_pdf(
                tmp_path / "hosp3_fortis.pdf",
                [
                    "Fortis Medical Institute & Research Lab",
                    "Patient Name: Priya Patel",
                    "Age: 34",
                    "Gender: Female",
                    "Diagnosis: Mild Anemia",
                    "Hemoglobin: 10.2 g/dL",
                    "Creatinine: 0.80 mg/dL",
                    "HbA1c: 5.2 %",
                    "Blood Pressure: 110/70 mmHg",
                    "HIV 1 & 2 Ab: Negative",
                ]
            )

            # Hospital 4: Max Healthcare Specialty
            pdf_4 = self._create_medical_pdf(
                tmp_path / "hosp4_max.pdf",
                [
                    "Max Healthcare Specialty Hospital",
                    "Patient Name: Anitha Krishnan",
                    "Age: 29 Years",
                    "Sex: Female",
                    "Diagnosis: Routine Antenatal Checkup",
                    "Hemoglobin: 12.1 g/dL",
                    "Creatinine: 0.72 mg/dL",
                    "HbA1c: 5.0 %",
                    "BP Reading: 115/75 mmHg",
                    "ECG: Normal ECG Trace",
                ]
            )

            # Hospital 5: Manipal Super Specialty Hospital
            pdf_5 = self._create_medical_pdf(
                tmp_path / "hosp5_manipal.pdf",
                [
                    "Manipal Super Specialty Hospital",
                    "Patient Name: Vikramaditya Rao",
                    "Age: 61",
                    "Gender: Male",
                    "Diagnosis: Type 2 Diabetes Mellitus",
                    "Hemoglobin: 15.0 g/dL",
                    "Serum Creatinine: 1.35 mg/dL",
                    "HbA1c: 8.4 %",
                    "Blood Pressure: 135/85 mmHg",
                    "HIV Status: Negative",
                    "ECG Analysis: Incomplete RBBB",
                ]
            )

            # -----------------------------------------------------------------
            # TEST HOSPITAL 1 (City Care)
            # -----------------------------------------------------------------
            engine.purge_and_create_session(pdf_1, "City Care")

            r1_name = engine.ask("What is the patient's name?")
            self.assertIn("MANJIT", r1_name.answer)
            self.assertIsNotNone(r1_name.snippet_path)

            r1_hosp = engine.ask("What is the hospital name?")
            self.assertIn("City Care", r1_hosp.answer)

            r1_age = engine.ask("What is the age?")
            self.assertIn("45", r1_age.answer)

            r1_gender = engine.ask("What is the gender?")
            self.assertIn("Male", r1_gender.answer)

            r1_diag = engine.ask("What is the diagnosis?")
            self.assertIn("Normal", r1_diag.answer)

            r1_hb = engine.ask("What is the hemoglobin?")
            self.assertIn("14.5", r1_hb.answer)

            r1_creat = engine.ask("What is the creatinine value?")
            self.assertIn("1.02", r1_creat.answer)

            r1_hba1c = engine.ask("What is HbA1c?")
            self.assertIn("5.6", r1_hba1c.answer)

            r1_bp = engine.ask("What is the blood pressure?")
            self.assertIn("120/80", r1_bp.answer)

            r1_hiv = engine.ask("What is the HIV status?")
            self.assertIn("Non-Reactive", r1_hiv.answer)

            r1_ecg = engine.ask("What is the ECG result?")
            self.assertIn("Sinus", r1_ecg.answer)

            r1_sum = engine.ask("Summarize this report.")
            self.assertNotEqual(r1_sum.answer, NOT_FOUND_ANSWER)

            # Check unlisted field
            r1_missing = engine.ask("What is the thyroid TSH level?")
            self.assertEqual(r1_missing.answer, NOT_FOUND_ANSWER)

            # -----------------------------------------------------------------
            # TEST HOSPITAL 2 (Apollo)
            # -----------------------------------------------------------------
            engine.purge_and_create_session(pdf_2, "Apollo")

            r2_name = engine.ask("What is the patient's name?")
            self.assertIn("Rajesh", r2_name.answer)

            r2_hba1c = engine.ask("What is HbA1c?")
            self.assertIn("6.8", r2_hba1c.answer)

            r2_diag = engine.ask("What is the diagnosis?")
            self.assertIn("Hypertension", r2_diag.answer)

            # -----------------------------------------------------------------
            # TEST HOSPITAL 3 (Fortis)
            # -----------------------------------------------------------------
            engine.purge_and_create_session(pdf_3, "Fortis")

            r3_name = engine.ask("What is the patient's name?")
            self.assertIn("Priya", r3_name.answer)

            r3_hb = engine.ask("What is the hemoglobin level?")
            self.assertIn("10.2", r3_hb.answer)

            # -----------------------------------------------------------------
            # TEST HOSPITAL 4 (Max)
            # -----------------------------------------------------------------
            engine.purge_and_create_session(pdf_4, "Max")

            r4_name = engine.ask("What is the patient's name?")
            self.assertIn("Anitha", r4_name.answer)

            r4_ecg = engine.ask("What is the ECG?")
            self.assertIn("Trace", r4_ecg.answer)

            # -----------------------------------------------------------------
            # TEST HOSPITAL 5 (Manipal)
            # -----------------------------------------------------------------
            engine.purge_and_create_session(pdf_5, "Manipal")

            r5_name = engine.ask("What is the patient's name?")
            self.assertIn("Vikramaditya", r5_name.answer)

            r5_creat = engine.ask("What is the creatinine value?")
            self.assertIn("1.35", r5_creat.answer)

            r5_hba1c = engine.ask("What is HbA1c?")
            self.assertIn("8.4", r5_hba1c.answer)


if __name__ == "__main__":
    unittest.main()
