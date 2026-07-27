"""
Google Gemini API Client for Document Understanding.
Sends Top 5 TF-IDF text blocks and natural language question to Gemini,
and returns structured JSON answers without hallucination.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
import requests

from pathlib import Path

logger = logging.getLogger("gemini_client")


def _load_env():
    """Automatically load environment variables from root .env file if present."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass


_load_env()

# GEMINI API Models to attempt in order of preference
GEMINI_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.0-flash"
]

NOT_FOUND_ANSWER = "The uploaded report does not contain this information."


class GeminiQAClient:
    """Interface for querying Google Gemini API with retrieved document text blocks."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY or GOOGLE_API_KEY found in environment. Gemini client will use fallback logic if invoked.")

    def test_connection(self) -> Dict[str, Any]:
        """Test Gemini API connection and return a successful response."""
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {
                "success": False,
                "status": "error",
                "message": "No GEMINI_API_KEY configured in environment or .env file."
            }

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": "Hello Gemini, return a 1-sentence status confirmation."}]}]
        }

        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    res_data = response.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = parts[0].get("text", "").strip() if parts else "Gemini API connected successfully."
                        return {
                            "success": True,
                            "status": "connected",
                            "model": model,
                            "response": text
                        }
                else:
                    logger.warning(f"Gemini API test ({model}) status {response.status_code}: {response.text[:200]}")
            except Exception as e:
                logger.warning(f"Gemini API test error ({model}): {e}")

        return {
            "success": False,
            "status": "failed",
            "message": "Unable to connect to Google Gemini API."
        }

    def summarize(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """
        Generate a concise, professional summary of document text blocks using Gemini API.

        Args:
            blocks (List[Dict[str, Any]]): Extracted text blocks with page_number and text.

        Returns:
            Optional[str]: Summary string or None if API call fails.
        """
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key or not blocks:
            return None

        formatted_blocks = []
        for idx, b in enumerate(blocks, 1):
            page_num = b.get("page_number", 1)
            text = b.get("text", "").strip()
            formatted_blocks.append(f"[Block {idx}] Page {page_num}:\n{text}")

        blocks_str = "\n\n".join(formatted_blocks)

        prompt = f"""You are an expert medical document understanding system.
Summarize the following document text blocks into a clear, professional 2-3 sentence clinical summary covering the patient, key laboratory findings, and overall diagnostic status.

Document Text Blocks:
{blocks_str}

Summary:"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 200
            }
        }

        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=12)
                if response.status_code == 200:
                    res_data = response.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            summary_text = parts[0].get("text", "").strip()
                            if summary_text:
                                logger.info(f"Gemini API ({model}) generated summary successfully.")
                                return summary_text
            except Exception as e:
                logger.warning(f"Error calling Gemini API for summary ({model}): {e}")

        return None

    def query(self, question: str, top_blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Send Top 5 text blocks and question to Gemini API and receive structured JSON response.

        Args:
            question (str): User question text.
            top_blocks (List[Dict[str, Any]]): Top 5 retrieved text blocks with page_number and text.

        Returns:
            Optional[Dict[str, Any]]: Dict with keys 'matched_text', 'page', 'answer', 'confidence' or None on failure.
        """
        if not self.api_key:
            logger.info("Skipping Gemini API call: API key not configured.")
            return None

        if not top_blocks:
            logger.info("No text blocks provided to Gemini client.")
            return None

        formatted_blocks = []
        for idx, b in enumerate(top_blocks, 1):
            page_num = b.get("page_number", 1)
            text = b.get("text", "").strip()
            formatted_blocks.append(f"[Block {idx}] (Page {page_num}):\n{text}")

        blocks_str = "\n\n".join(formatted_blocks)

        prompt = f"""You are an expert medical document understanding system.
Your task is to answer the user's question using ONLY the provided text blocks extracted from a document.

Rules:
1. Base your answer strictly on the provided text blocks below. DO NOT guess or use outside knowledge.
2. If the requested information is not present in the provided text blocks, return JSON:
   {{"matched_text": "", "page": null, "answer": "The uploaded report does not contain this information.", "confidence": 0.0}}
3. If found, return STRICT JSON with exactly these 4 fields:
   - "matched_text": string containing the exact line or text segment from the block that contains the target field or value (e.g. "MANJIT SINGH" or "Patient Name: MANJIT SINGH" or "Creatinine: 1.02 mg/dL")
   - "page": integer page number where the answer was found
   - "answer": concise answer string (e.g. "MANJIT SINGH" or "1.02 mg/dL")
   - "confidence": float between 0.0 and 1.0 (e.g. 0.98)
4. Output STRICT JSON ONLY. Do NOT wrap in markdown, code blocks, or extra text.

Document Text Blocks:
{blocks_str}

User Question: {question}
"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.9,
                "maxOutputTokens": 300,
                "responseMimeType": "application/json"
            }
        }

        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.api_key}"
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=12)
                if response.status_code == 200:
                    res_data = response.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if content_parts:
                            raw_text = content_parts[0].get("text", "").strip()
                            parsed_json = self._parse_json_response(raw_text)
                            if parsed_json:
                                logger.info(f"Gemini API ({model}) returned valid answer: {parsed_json.get('answer')}")
                                return parsed_json
                else:
                    logger.warning(f"Gemini API ({model}) returned status code {response.status_code}: {response.text[:200]}")
            except Exception as e:
                logger.warning(f"Error calling Gemini API ({model}): {e}")

        logger.warning("All Gemini API models failed or returned non-JSON. Falling back to local matcher.")
        return None

    @staticmethod
    def _parse_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
        """Parse raw text response from Gemini into a structured JSON dict."""
        if not raw_text:
            return None

        # Remove markdown code blocks if present (e.g. ```json ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Extract JSON object substring if preamble text is present
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                answer = str(data.get("answer", "")).strip()
                matched_text = str(data.get("matched_text", "")).strip()
                page = data.get("page")
                if page is not None:
                    try:
                        page = int(page)
                    except (ValueError, TypeError):
                        page = None
                
                try:
                    confidence = float(data.get("confidence", 0.95))
                except (ValueError, TypeError):
                    confidence = 0.95

                return {
                    "matched_text": matched_text,
                    "page": page,
                    "answer": answer,
                    "confidence": round(confidence, 2)
                }
        except Exception as e:
            logger.warning(f"Failed to parse Gemini JSON output: {e}. Raw: '{raw_text}'")

        return None
