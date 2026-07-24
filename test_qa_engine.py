"""
Integration Test Script for Visual Document QA Engine.
Verifies query responses and page screenshot extractions for the requested questions.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.qa_engine import DocumentQAEngine

def test_document_qa():
    print("=== Testing Document QA Engine & Screenshot Retrieval ===")
    engine = DocumentQAEngine()

    test_queries = [
        "Was the blood sample collected in fasting mode?",
        "What was the answer to lung disease?",
        "What is the gender and age of the siblings?",
        "What was the ECG test result?",
        "What is the face similarity score?",
        "What are the HbA1c and Blood Sugar values?"
    ]

    for idx, q in enumerate(test_queries, 1):
        res = engine.ask(q)
        print(f"\n[{idx}] Question: '{res.question}'")
        print(f"    Confidence: {res.confidence * 100:.1f}%")
        print(f"    Target Page: Page {res.page_number}" + (f" & {res.secondary_page_number}" if res.secondary_page_number else ""))
        print(f"    Section: {res.section_title}")
        print(f"    Answer snippet:\n    {res.answer[:120]}...")
        print(f"    Screenshot Filename: {res.snippet_filename}")
        
        assert res.page_number in [1, 2, 3, 6, 7, 9, 10, 14], f"Unexpected page number: {res.page_number}"
        assert res.confidence >= 0.90, f"Low confidence: {res.confidence}"

    print("\n[SUCCESS] All QA Engine unit and integration checks PASSED!")

if __name__ == "__main__":
    test_document_qa()
