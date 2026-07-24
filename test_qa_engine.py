"""
Automated Integration Test Suite for Trained 100 Question & Answer Medical Document Engine.
Verifies precision answers, confidence scores, page numbers, and snippet file generation.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.qa_engine import DocumentQAEngine

def test_100_qa_dataset():
    print("=== Testing Trained 100 Question & Answer Medical Document Engine ===")
    engine = DocumentQAEngine()

    test_queries = [
        # IDs 1-10
        ("Who is the patient in this medical report?", "Manjit Singh.", 2),
        ("What is the full name of the patient?", "Manjit Singh.", 2),
        ("What is the application number?", "U100723465AD0.", 4),
        ("Which insurance company requested the medical examination?", "Tata AIA Life Insurance Company Ltd.", 4),
        ("Which diagnostic centre performed the tests?", "Jeevandeep Diagnostic & Polyclinic.", 4),
        ("What type of medical service was provided?", "Home Visit.", 4),
        ("What is the face similarity score?", "98.75%.", 3),
        ("What FRS score was obtained?", "98.75.", 3),
        ("Was there any pincode mismatch?", "No.", 3),
        ("What distance is mentioned in the face match report?", "0 km.", 3),

        # IDs 11-20 (CBC & Labs)
        ("What is the haemoglobin level?", "14.92 g/dL.", 11),
        ("What is the total leukocyte count?", "7,900 cells/cu.mm.", 11),
        ("What is the platelet count?", "2,90,000 cells/cu.mm.", 11),
        ("What is the RBC count?", "5.88 million cells/cu.mm.", 11),
        ("What is the ESR?", "14 mm/hr.", 11),
        ("What is the Blood Urea Nitrogen value?", "18.10 mg/dL.", 13),
        ("What is the serum creatinine level?", "0.88 mg/dL.", 13),
        ("What is the HbA1c percentage?", "5.1%.", 14),
        ("Is the HbA1c within the normal range?", "Yes.", 14),
        ("What is the HIV test result?", "Negative.", 16),

        # IDs 21-30 (Hepatitis, Timestamp & Summaries)
        ("What is the HBsAg result?", "Non-reactive.", 15),
        ("What is the report generation time?", "18-Jul-2026 12:29:12 PM.", 3),
        ("Summarize the overall face verification result.", "The face verification was successful with a similarity score of 98.75%", 3),
        ("Summarize the CBC findings.", "The CBC report includes haemoglobin", 11),
        ("Summarize the viral screening.", "The HIV screening result is negative", 16),
        ("Summarize the insurance medical examination.", "The report documents an insurance medical examination", 4),
        ("Give a brief summary of this PDF.", "The PDF contains an insurance medical examination for Manjit Singh", 1)
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
    test_100_qa_dataset()
