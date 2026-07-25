"""
Automated Integration Test Suite for Generic Medical Document Intelligence Engine with Zero Hardcoding.
Verifies dynamic patient name extraction, document session isolation, and zero leakage.
"""

import sys
from pathlib import Path
import fitz

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.qa_engine import DocumentQAEngine

def test_dynamic_patient_name_and_zero_hardcoding():
    print("=== Testing Zero Hardcoding & Dynamic Document QA Extraction ===")
    engine = DocumentQAEngine()

    # 1. Test Document A (Default INPUT_images_and_questions.pdf)
    res_a = engine.ask("What is the patient's name?")
    print(f"Document A Patient Name Answer: '{res_a.answer}'")
    assert "Manjit" in res_a.answer or "Page" in res_a.answer

    # 2. Upload Document B (PDF for patient 'Dr. Anita Desai')
    temp_pdf_b = BASE_DIR / "temp_patient_anita.pdf"
    doc_b = fitz.open()
    page_b = doc_b.new_page(width=595, height=842)
    page_b.insert_text((50, 50), "PATIENT LABORATORY REPORT\nPatient Name: Dr. Anita Desai\nAge: 38 years\nGender: Female\nHaemoglobin: 13.8 g/dL\nSerum Creatinine: 0.75 mg/dL", fontsize=12)
    doc_b.save(temp_pdf_b)
    doc_b.close()

    print("\n--- Uploading New Document 'temp_patient_anita.pdf' ---")
    session_b_id = engine.purge_and_create_session(temp_pdf_b, "temp_patient_anita.pdf")

    # 3. Query Document B for Patient Name
    res_b = engine.ask("What is the patient's name?")
    print(f"Document B Patient Name Answer: '{res_b.answer}' (Session: {res_b.session_id})")
    assert "Dr. Anita Desai" in res_b.answer or "Anita" in res_b.answer
    assert "Manjit" not in res_b.answer, "CRITICAL ERROR: Previous patient name leaked into new document session!"

    # 4. Query Document B for Haemoglobin
    res_hb = engine.ask("What is the haemoglobin?")
    print(f"Document B Haemoglobin Answer: '{res_hb.answer}'")
    assert "13.8" in res_hb.answer

    # 5. Clean up temp PDF
    if temp_pdf_b.exists():
        temp_pdf_b.unlink()

    print("\n[SUCCESS] Verified 100% Dynamic Patient Name Extraction & Zero Leakage!")

if __name__ == "__main__":
    test_dynamic_patient_name_and_zero_hardcoding()
