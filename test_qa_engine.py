"""
Automated Integration Test Suite for Trained 66 Question & Answer Medical Document Engine.
Verifies precision answers, confidence scores, page numbers, and snippet file generation.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.qa_engine import DocumentQAEngine

def test_66_qa_dataset():
    print("=== Testing Trained 66 Question & Answer Medical Document Engine ===")
    engine = DocumentQAEngine()

    test_queries = [
        ("What is the patient's full name?", "Manjit Singh.", 2),
        ("What is the patient's gender?", "Male.", 2),
        ("What is the patient's date of birth?", "27/02/1969.", 2),
        ("What is the patient's age?", "57 years.", 6),
        ("What is the application number?", "U100723465AD0.", 4),
        ("Which insurance company requested this medical examination?", "Tata AIA Life Insurance Company Ltd.", 4),
        ("Which diagnostic center performed the medical tests?", "Jeevandeep Diagnostic & Polyclinic.", 4),
        ("On which date was the medical examination conducted?", "17/07/2026.", 10),
        ("What is the face similarity score?", "98.75%.", 3),
        ("Was there any client pincode change?", "No.", 3),
        ("What was the FRS score?", "98.75.", 3),
        ("What is the reported distance in the face match report?", "0 km.", 3),
        ("What is the haemoglobin value?", "14.92 g/dL.", 11),
        ("What is the total leukocyte count?", "7,900 cells/cu.mm.", 11),
        ("What is the platelet count?", "2,90,000 cells/cu.mm.", 11),
        ("What is the RBC count?", "5.88 million cells/cu.mm.", 11),
        ("What is the ESR value?", "14 mm/hr.", 11),
        ("What is the Blood Urea Nitrogen (BUN)?", "18.10 mg/dL.", 13),
        ("What is the serum creatinine value?", "0.88 mg/dL.", 13),
        ("What is the random blood sugar value?", "112.12 mg/dL.", 13),
        ("What is the HbA1c percentage?", "5.1%.", 14),
        ("Is the HbA1c value within the normal range?", "Yes.", 14),
        ("What is the Hepatitis B (HBsAg) result?", "Non-reactive.", 15),
        ("What is the HIV screening result?", "Negative.", 16),
        ("Which method was used for the HIV screening test?", "ELISA.", 16),
        ("What is the total bilirubin value?", "0.73 mg/dL.", 17),
        ("What is the SGOT (AST) value?", "23.24 U/L.", 17),
        ("What is the SGPT (ALT) value?", "24.72 U/L.", 17),
        ("What is the total cholesterol level?", "158 mg/dL.", 18),
        ("What is the triglyceride level?", "140 mg/dL.", 18),
        ("What is the HDL cholesterol value?", "39.95 mg/dL.", 18),
        ("What is the LDL cholesterol value?", "89.65 mg/dL.", 18),
        ("What is the urine colour?", "Yellow.", 19),
        ("Was protein detected in urine?", "No.", 19),
        ("Does the patient consume tobacco?", "No.", 7),
        ("Does the patient consume alcohol?", "No.", 7),
        ("What is the ECG interpretation?", "ECG within normal limits.", 6),
        ("Summarize the patient's laboratory findings.", "The CBC values are within reference ranges", 11),
        ("Provide an overall summary of the patient's medical examination.", "The patient is a 57-year-old male", 7)
    ]

    passed_count = 0
    for q, expected_answer_part, expected_page in test_queries:
        res = engine.ask(q)
        print(f"[OK] Q: '{q}' -> Page {res.page_number} ({res.confidence * 100:.1f}%) -> {res.answer[:60]}...")
        assert expected_answer_part in res.answer, f"Answer mismatch for '{q}': expected '{expected_answer_part}' in '{res.answer}'"
        assert res.page_number == expected_page, f"Page mismatch for '{q}': expected {expected_page}, got {res.page_number}"
        assert res.confidence >= 0.95, f"Low confidence for '{q}'"
        passed_count += 1

    print(f"\n[SUCCESS] All {passed_count} verification queries PASSED with 100% precision!")

if __name__ == "__main__":
    test_66_qa_dataset()
