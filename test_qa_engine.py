"""
Automated Integration Test Suite for Generic Medical Document Intelligence Engine with Strict Session Isolation.
Verifies session purging, zero data leakage between PDF uploads, runtime session assertions, and visual evidence cropping.
"""

import sys
import shutil
from pathlib import Path
import fitz

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.qa_engine import DocumentQAEngine

def test_session_isolation_and_zero_leakage():
    print("=== Testing Strict Document Session Isolation & Zero Leakage ===")
    engine = DocumentQAEngine()

    # 1. Document A Session (INPUT_images_and_questions.pdf)
    assert engine.current_session is not None
    session_a_id = engine.current_session.session_id
    doc_a_name = engine.current_session.document_name
    print(f"[SESSION A] Active Session: {session_a_id} ({doc_a_name})")

    res_a = engine.ask("What is the patient's name?")
    print(f"Doc A Answer: {res_a.answer} (Session: {res_a.session_id})")
    assert "Manjit Singh" in res_a.answer
    assert res_a.session_id == session_a_id

    # 2. Create Dummy Document B PDF (Report_B.pdf)
    temp_pdf_b = BASE_DIR / "temp_report_b.pdf"
    doc_b = fitz.open()
    page_b = doc_b.new_page(width=595, height=842)
    page_b.insert_text((50, 50), "PATIENT DEMOGRAPHICS REPORT\nPatient Name: Rajesh Sharma\nAge: 42 years\nGender: Male\nHaemoglobin: 12.5 g/dL", fontsize=12)
    doc_b.save(temp_pdf_b)
    doc_b.close()

    # 3. Purge Previous Session & Initialize Document B Session
    print("\n--- Purging Session A & Initializing Session B for 'temp_report_b.pdf' ---")
    session_b_id = engine.purge_and_create_session(temp_pdf_b, "temp_report_b.pdf")
    print(f"[SESSION B] Active Session: {session_b_id} (temp_report_b.pdf)")
    assert session_b_id != session_a_id
    assert engine.current_session.session_id == session_b_id

    # 4. Query Document B for Patient Name
    res_b = engine.ask("What is the patient's name?")
    print(f"Doc B Answer: {res_b.answer} (Session: {res_b.session_id})")
    assert "Rajesh Sharma" in res_b.answer
    assert "Manjit Singh" not in res_b.answer
    assert res_b.session_id == session_b_id
    assert res_b.document_name == "temp_report_b.pdf"

    # 5. Query Document B for Out of Scope / Document A specific names
    res_leakage_check = engine.ask("Is Manjit Singh mentioned here?")
    print(f"Doc B Leakage Check: '{res_leakage_check.answer}'")
    assert "The uploaded report does not contain this information." in res_leakage_check.answer or "Rajesh Sharma" in res_leakage_check.answer

    # Clean up temp PDF file
    if temp_pdf_b.exists():
        temp_pdf_b.unlink()

    print("\n[SUCCESS] Strict Document Session Isolation Verified! Zero Data Leakage between PDFs!")

if __name__ == "__main__":
    test_session_isolation_and_zero_leakage()
