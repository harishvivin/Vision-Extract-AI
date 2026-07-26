"""
Automated Integration Test Suite for the generic document QA engine.
Verifies generic session creation, absence of stale medical-specific internals, and expected behavior for present and absent queries.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.qa_engine import DocumentQAEngine

NOT_FOUND_MESSAGE = "The uploaded PDF does not contain this information."


def test_generic_qa_engine():
    print("=== Testing generic document QA engine ===")
    engine = DocumentQAEngine()

    assert engine.current_session is None, "Engine initialization failed: current_session must be None without preloads!"
    print("[PASS] No default session on engine initialization.")

    assert not hasattr(engine, "concept_aliases"), "Architecture Error: concept_aliases must NOT exist in qa_engine!"
    print("[PASS] No stale concept_aliases attribute present.")

    default_pdf = BASE_DIR / "INPUT_images_and_questions.pdf"
    if default_pdf.exists():
        session_id = engine.purge_and_create_session(default_pdf, default_pdf.name)
        assert engine.current_session is not None, "Session creation failed!"
        assert engine.current_session.session_id == session_id, "Session ID mismatch after creation."
        assert engine.current_session.indexed_blocks, "No text blocks were indexed from the PDF."
        print(f"[PASS] Session {session_id[:8]} created and indexed {len(engine.current_session.indexed_blocks)} text blocks.")

        present_question = "What is the application number?"
        present_result = engine.ask(present_question)
        print(f"[OK] Present query returned page {present_result.page_number} and answer '{present_result.answer[:80]}'.")
        assert present_result.answer and present_result.answer != NOT_FOUND_MESSAGE, "Expected a non-empty answer for a present query."
        assert present_result.page_number is not None, "Expected a page number for a present query."

        absent_question = "What is the car insurance premium?"
        absent_result = engine.ask(absent_question)
        print(f"[OK] Absent query returned answer '{absent_result.answer}'.")
        assert absent_result.answer == NOT_FOUND_MESSAGE, "Expected the not-found message for an absent query."
        assert absent_result.page_number is None, "Expected no page number for an absent query."
        assert absent_result.snippet_path is None, "Expected no snippet for an absent query."

    else:
        print("[SKIP] Test PDF not found; skipping session-based QA validation.")


if __name__ == "__main__":
    test_generic_qa_engine()
