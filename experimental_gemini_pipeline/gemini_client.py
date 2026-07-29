"""
Gemini API Client for Experimental PDF Document Localization.
Supports dual API keys (primary & fallback) with transparent automatic failover,
Flash-Lite model selection, temperature=0.0, and strict JSON output parsing.
"""

import os
import json
import logging
import re
import requests
from pathlib import Path
from typing import Dict, Any, Optional

from .config import (
    GEMINI_API_KEY_PRIMARY,
    GEMINI_API_KEY_FALLBACK,
    GEMINI_MODELS,
    DEFAULT_TEMPERATURE
)

logger = logging.getLogger("experimental_gemini_client")
logging.basicConfig(level=logging.INFO)


class ExperimentalGeminiClient:
    """
    Experimental Gemini API client dedicated to PDF spatial localization.
    Features transparent failover between primary and fallback API keys.
    """

    def __init__(self, primary_key: Optional[str] = None, fallback_key: Optional[str] = None):
        self.primary_key = primary_key or GEMINI_API_KEY_PRIMARY
        self.fallback_key = fallback_key or GEMINI_API_KEY_FALLBACK or self.primary_key

        if not self.primary_key:
            logger.warning("No GEMINI_API_KEY_PRIMARY configured.")

    def query_pdf(self, pdf_path: str | Path, question: str) -> Dict[str, Any]:
        """
        Query Gemini API with PDF document and localization prompt.
        Transparently retries with fallback key if any API exception occurs.

        Args:
            pdf_path (str | Path): Path to PDF document.
            question (str): Natural language question or field name.

        Returns:
            Dict[str, Any]: Parsed JSON response containing localization data.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

        from .prompt_builder import build_prompt
        prompt_text = build_prompt(question)

        # 1. Attempt using Primary API Key
        try:
            logger.info("Attempting PDF query using PRIMARY API Key...")
            return self._execute_call(pdf_path, prompt_text, api_key=self.primary_key)
        except Exception as e_primary:
            logger.warning(f"PRIMARY API Key failed (Error: {e_primary}). Retrying transparently with FALLBACK API Key...")

        # 2. Transparent Failover: Attempt using Fallback API Key
        try:
            logger.info("Attempting PDF query using FALLBACK API Key...")
            return self._execute_call(pdf_path, prompt_text, api_key=self.fallback_key)
        except Exception as e_fallback:
            logger.error(f"Both PRIMARY and FALLBACK API Keys failed. Fallback error: {e_fallback}")
            return {
                "found": False,
                "error": f"API Failover Exhausted: Primary ({e_primary}), Fallback ({e_fallback})"
            }

    def _execute_call(self, pdf_path: Path, prompt_text: str, api_key: str) -> Dict[str, Any]:
        """
        Internal worker to upload PDF bytes or media and invoke Gemini generateContent.
        Raises Exception on any API failure so caller can trigger transparent retry.
        """
        if not api_key:
            raise ValueError("API Key is missing or invalid.")

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # Step A: Upload PDF to Gemini Files API
        upload_url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
        metadata = json.dumps({"file": {"display_name": pdf_path.name}}).encode("utf-8")

        boundary = "===GEMINI_PDF_UPLOAD_BOUNDARY==="
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b"Content-Type: application/json; charset=UTF-8\r\n\r\n")
        body.extend(metadata)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b"Content-Type: application/pdf\r\n\r\n")
        body.extend(pdf_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        upload_headers = {
            "Content-Type": f"multipart/related; boundary={boundary}"
        }

        up_response = requests.post(upload_url, headers=upload_headers, data=bytes(body), timeout=30)
        if up_response.status_code != 200:
            raise RuntimeError(f"Gemini File Upload API failed with status {up_response.status_code}: {up_response.text}")

        file_info = up_response.json().get("file", {})
        file_uri = file_info.get("uri")
        if not file_uri:
            raise RuntimeError(f"Gemini File Upload API did not return a valid file URI: {up_response.text}")

        # Step B: Generate Content using Flash-Lite models
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "application/pdf",
                                "fileUri": file_uri
                            }
                        },
                        {
                            "text": prompt_text
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": DEFAULT_TEMPERATURE,
                "responseMimeType": "application/json"
            }
        }

        last_error = None
        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            gen_headers = {"Content-Type": "application/json"}
            try:
                response = requests.post(url, headers=gen_headers, json=payload, timeout=25)
                if response.status_code == 200:
                    res_data = response.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_text = parts[0].get("text", "").strip()
                            parsed_json = self._parse_json(raw_text)
                            if parsed_json is not None:
                                return parsed_json
                else:
                    last_error = f"Model {model} HTTP {response.status_code}: {response.text}"
            except Exception as e:
                last_error = f"Model {model} exception: {e}"

        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    @staticmethod
    def _parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
        """Parse raw JSON output from Gemini model."""
        if not raw_text:
            return None

        # Clean markdown code blocks if present
        codeblock_match = re.search(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", raw_text, re.IGNORECASE)
        text_to_parse = codeblock_match.group(1).strip() if codeblock_match else raw_text.strip()

        # Match first JSON object
        json_match = re.search(r"\{[\s\S]*\}", text_to_parse)
        if json_match:
            text_to_parse = json_match.group(0)

        try:
            data = json.loads(text_to_parse)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        return None
