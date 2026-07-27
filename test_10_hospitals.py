"""
Comprehensive 10-Hospital Report RAG System Test Suite.
Verifies RAG accuracy, zero hallucination, page location, and evidence cropping across 10 distinct hospital report formats.
Tests 13 mandatory medical fields: Patient Name, Age, Gender, Diagnosis, Hospital, Doctor,
Creatinine, HbA1c, Hemoglobin, Blood Pressure, ECG, HIV, Summary.
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


class TenHospitalsBenchmarkTest(unittest.TestCase):

    def _create_medical_pdf(self, file_path: Path, text_lines: list[str]) -> Path:
        """Helper to generate synthetic multi-line medical report PDF."""
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        y_offset = 40
        for line in text_lines:
            page.insert_text((40, y_offset), line, fontsize=11)
            y_offset += 24
        doc.save(file_path)
        doc.close()
        return file_path

    def test_10_hospital_formats_rag_accuracy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            outputs_path = tmp_path / "qa_outputs"
            engine = DocumentQAEngine(outputs_dir=outputs_path)

            # -----------------------------------------------------------------
            # 1. City Care General Hospital
            # -----------------------------------------------------------------
            pdf_1 = self._create_medical_pdf(
                tmp_path / "hosp1_city_care.pdf",
                [
                    "City Care General Hospital - Laboratory Report",
                    "Patient Name: MANJIT SINGH | Age: 45 Years | Gender: Male",
                    "Attending Doctor: Dr. Ramesh Verma, MD",
                    "Clinical Impression: Normal Baseline Health Screening",
                    "Hemoglobin: 14.5 g/dL",
                    "Serum Creatinine: 1.02 mg/dL",
                    "HbA1c: 5.6 %",
                    "Blood Pressure: 120/80 mmHg",
                    "HIV Screening Test: Non-Reactive",
                    "ECG Result: Normal Sinus Rhythm",
                ]
            )

            # -----------------------------------------------------------------
            # 2. Apollo Health Diagnostics
            # -----------------------------------------------------------------
            pdf_2 = self._create_medical_pdf(
                tmp_path / "hosp2_apollo.pdf",
                [
                    "Apollo Health Diagnostics Center",
                    "Name of Patient: Rajesh Sharma",
                    "Age: 52 | Sex: Male",
                    "Consultant Doctor: Dr. Sunita Kapoor",
                    "Primary Diagnosis: Essential Hypertension",
                    "Hemoglobin Level: 13.8 g/dL",
                    "Creatinine: 0.95 mg/dL",
                    "Glycated Hemoglobin (HbA1c): 6.8 %",
                    "BP: 140/90 mmHg",
                    "HIV Status: Negative",
                    "ECG: Sinus Bradycardia",
                ]
            )

            # -----------------------------------------------------------------
            # 3. Fortis Medical Institute
            # -----------------------------------------------------------------
            pdf_3 = self._create_medical_pdf(
                tmp_path / "hosp3_fortis.pdf",
                [
                    "Fortis Medical Institute & Research Lab",
                    "Patient Name: Priya Patel",
                    "Age: 34 Years | Gender: Female",
                    "Ref. Physician: Dr. Anil Deshmukh",
                    "Diagnosis: Mild Anemia",
                    "Hemoglobin: 10.2 g/dL",
                    "Creatinine: 0.80 mg/dL",
                    "HbA1c: 5.2 %",
                    "Blood Pressure: 110/70 mmHg",
                    "HIV 1 & 2 Ab: Negative",
                    "ECG Trace: Normal Baseline",
                ]
            )

            # -----------------------------------------------------------------
            # 4. Max Healthcare Specialty
            # -----------------------------------------------------------------
            pdf_4 = self._create_medical_pdf(
                tmp_path / "hosp4_max.pdf",
                [
                    "Max Healthcare Specialty Hospital",
                    "Patient Name: Anitha Krishnan",
                    "Age: 29 Years | Sex: Female",
                    "Doctor: Dr. Meenakshi Sundaram",
                    "Diagnosis: Routine Antenatal Checkup",
                    "Hemoglobin: 12.1 g/dL",
                    "Creatinine: 0.72 mg/dL",
                    "HbA1c: 5.0 %",
                    "BP Reading: 115/75 mmHg",
                    "HIV Test: Non-Reactive",
                    "ECG: Normal ECG Trace",
                ]
            )

            # -----------------------------------------------------------------
            # 5. Manipal Super Specialty Hospital
            # -----------------------------------------------------------------
            pdf_5 = self._create_medical_pdf(
                tmp_path / "hosp5_manipal.pdf",
                [
                    "Manipal Super Specialty Hospital",
                    "Patient Name: Vikramaditya Rao",
                    "Age: 61 | Gender: Male",
                    "Physician: Dr. Suresh Kulkarni",
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
            # 6. Medanta Heart & Diagnostic Institute
            # -----------------------------------------------------------------
            pdf_6 = self._create_medical_pdf(
                tmp_path / "hosp6_medanta.pdf",
                [
                    "Medanta Heart & Diagnostic Institute",
                    "Patient Name: Gurpreet Singh",
                    "Age: 58 Years | Sex: Male",
                    "Doctor: Dr. Naresh Trehan",
                    "Diagnosis: Coronary Artery Disease Evaluation",
                    "Hemoglobin: 14.1 g/dL",
                    "Creatinine: 1.10 mg/dL",
                    "HbA1c: 6.1 %",
                    "Blood Pressure: 130/82 mmHg",
                    "HIV Screening: Negative",
                    "ECG: T-Wave Inversion in V4-V6",
                ]
            )

            # -----------------------------------------------------------------
            # 7. Columbia Asia Referral Hospital
            # -----------------------------------------------------------------
            pdf_7 = self._create_medical_pdf(
                tmp_path / "hosp7_columbia.pdf",
                [
                    "Columbia Asia Referral Hospital",
                    "Name: Lakshmi Narayanan",
                    "Age: 67 | Gender: Female",
                    "Doctor: Dr. Radhika Shenoy",
                    "Diagnosis: Chronic Kidney Disease Stage 2",
                    "Hemoglobin: 11.0 g/dL",
                    "Creatinine: 1.85 mg/dL",
                    "HbA1c: 7.2 %",
                    "Blood Pressure: 145/92 mmHg",
                    "HIV 1 & 2: Non-Reactive",
                    "ECG: Left Ventricular Hypertrophy",
                ]
            )

            # -----------------------------------------------------------------
            # 8. Ruby Hall Clinic Diagnostics
            # -----------------------------------------------------------------
            pdf_8 = self._create_medical_pdf(
                tmp_path / "hosp8_ruby_hall.pdf",
                [
                    "Ruby Hall Clinic Diagnostics & Research",
                    "Patient Name: Rohan Mehta",
                    "Age: 38 Years | Sex: Male",
                    "Consultant: Dr. PK Grant",
                    "Diagnosis: Acute Gastritis",
                    "Hemoglobin: 15.2 g/dL",
                    "Creatinine: 0.90 mg/dL",
                    "HbA1c: 5.4 %",
                    "Blood Pressure: 118/76 mmHg",
                    "HIV Test: Negative",
                    "ECG: Normal Rhythm",
                ]
            )

            # -----------------------------------------------------------------
            # 9. Care Hospitals Pathology Lab
            # -----------------------------------------------------------------
            pdf_9 = self._create_medical_pdf(
                tmp_path / "hosp9_care.pdf",
                [
                    "Care Hospitals Pathology Lab",
                    "Patient Name: Kavitha Reddy",
                    "Age: 42 | Gender: Female",
                    "Doctor: Dr. Srinivas Rao",
                    "Diagnosis: Hypothyroidism",
                    "Hemoglobin: 12.8 g/dL",
                    "Creatinine: 0.78 mg/dL",
                    "HbA1c: 5.9 %",
                    "Blood Pressure: 122/80 mmHg",
                    "HIV Status: Non-Reactive",
                    "ECG: Normal Sinus Rhythm",
                ]
            )

            # -----------------------------------------------------------------
            # 10. KIMS Hospitals & Research Centre
            # -----------------------------------------------------------------
            pdf_10 = self._create_medical_pdf(
                tmp_path / "hosp10_kims.pdf",
                [
                    "KIMS Hospitals & Research Centre",
                    "Patient Name: Mohammed Ibrahim",
                    "Age: 50 Years | Sex: Male",
                    "Doctor: Dr. B. Bhaskar Rao",
                    "Diagnosis: Metabolic Syndrome",
                    "Hemoglobin: 14.7 g/dL",
                    "Creatinine: 1.05 mg/dL",
                    "HbA1c: 6.6 %",
                    "Blood Pressure: 138/88 mmHg",
                    "HIV 1 & 2 Ab: Negative",
                    "ECG: Sinus Rhythm with PACs",
                ]
            )

            # -----------------------------------------------------------------
            # RUN BENCHMARK TESTS ACROSS ALL 10 HOSPITALS & ALL 13 QUESTIONS
            # -----------------------------------------------------------------
            hospital_tests = [
                (pdf_1, "City Care", "MANJIT", "45", "Male", "Normal", "Ramesh", "1.02", "5.6", "14.5", "120/80", "Sinus", "Non-Reactive"),
                (pdf_2, "Apollo", "Rajesh", "52", "Male", "Hypertension", "Sunita", "0.95", "6.8", "13.8", "140/90", "Bradycardia", "Negative"),
                (pdf_3, "Fortis", "Priya", "34", "Female", "Anemia", "Anil", "0.80", "5.2", "10.2", "110/70", "Normal", "Negative"),
                (pdf_4, "Max", "Anitha", "29", "Female", "Antenatal", "Meenakshi", "0.72", "5.0", "12.1", "115/75", "Trace", "Non-Reactive"),
                (pdf_5, "Manipal", "Vikramaditya", "61", "Male", "Diabetes", "Suresh", "1.35", "8.4", "15.0", "135/85", "RBBB", "Negative"),
                (pdf_6, "Medanta", "Gurpreet", "58", "Male", "Coronary", "Trehan", "1.10", "6.1", "14.1", "130/82", "Inversion", "Negative"),
                (pdf_7, "Columbia", "Lakshmi", "67", "Female", "Kidney", "Radhika", "1.85", "7.2", "11.0", "145/92", "Hypertrophy", "Non-Reactive"),
                (pdf_8, "Ruby Hall", "Rohan", "38", "Male", "Gastritis", "Grant", "0.90", "5.4", "15.2", "118/76", "Normal", "Negative"),
                (pdf_9, "Care", "Kavitha", "42", "Female", "Hypothyroidism", "Srinivas", "0.78", "5.9", "12.8", "122/80", "Normal", "Non-Reactive"),
                (pdf_10, "KIMS", "Ibrahim", "50", "Male", "Metabolic", "Bhaskar", "1.05", "6.6", "14.7", "138/88", "PACs", "Negative"),
            ]

            for pdf_file, hosp_keyword, name, age, gender, diag, doc, creat, hba1c, hb, bp, ecg, hiv in hospital_tests:
                # Session Purge & Fresh Session Creation
                session_id = engine.purge_and_create_session(pdf_file, pdf_file.name)
                self.assertIsNotNone(session_id)

                # 1. Patient Name
                r_name = engine.ask("What is the patient's name?")
                self.assertIn(name, r_name.answer)
                self.assertEqual(r_name.page_number, 1)
                self.assertIsNotNone(r_name.snippet_path)
                self.assertTrue(Path(r_name.snippet_path).exists())

                # 2. Age
                r_age = engine.ask("What is the patient's age?")
                self.assertIn(age, r_age.answer)
                self.assertEqual(r_age.page_number, 1)

                # 3. Gender
                r_gender = engine.ask("What is the gender?")
                self.assertIn(gender, r_gender.answer)

                # 4. Diagnosis
                r_diag = engine.ask("What is the diagnosis?")
                self.assertIn(diag, r_diag.answer)

                # 5. Hospital
                r_hosp = engine.ask("What is the hospital or lab name?")
                self.assertIn(hosp_keyword, r_hosp.answer)

                # 6. Doctor
                r_doc = engine.ask("What is the attending doctor's name?")
                self.assertIn(doc, r_doc.answer)

                # 7. Creatinine
                r_creat = engine.ask("What is the creatinine value?")
                self.assertIn(creat, r_creat.answer)

                # 8. HbA1c
                r_hba1c = engine.ask("What is the HbA1c percentage?")
                self.assertIn(hba1c, r_hba1c.answer)

                # 9. Hemoglobin
                r_hb = engine.ask("What is the hemoglobin level?")
                self.assertIn(hb, r_hb.answer)

                # 10. Blood Pressure
                r_bp = engine.ask("What is the blood pressure reading?")
                self.assertIn(bp, r_bp.answer)

                # 11. ECG
                r_ecg = engine.ask("What is the ECG result?")
                self.assertIn(ecg, r_ecg.answer)

                # 12. HIV
                r_hiv = engine.ask("What is the HIV status?")
                self.assertIn(hiv, r_hiv.answer)

                # 13. Summary
                r_sum = engine.ask("Summarize this medical report.")
                self.assertNotEqual(r_sum.answer, NOT_FOUND_ANSWER)
                self.assertGreater(len(r_sum.answer), 15)

                # Zero Hallucination Test (unlisted parameter)
                r_missing = engine.ask("What is the thyroid TSH level?")
                self.assertEqual(r_missing.answer, NOT_FOUND_ANSWER)
                self.assertEqual(r_missing.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
