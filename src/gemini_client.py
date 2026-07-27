"""
Google Gemini RAG API Client for Medical Document Understanding.
Sends Top 5 retrieved document chunks and natural language questions to Gemini,
returning strict structured JSON answers without hallucination.
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
    """Interface for querying Google Gemini API with Top 5 retrieved document chunks."""

    _KEY_INVALID: bool = False

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            logger.warning("No GEMINI_API_KEY found in environment. Gemini client will use local RAG matcher fallback.")

    def test_connection(self) -> Dict[str, Any]:
        """Test Gemini API connection and return status confirmation."""
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return {
                "success": False,
                "status": "error",
                "message": "No GEMINI_API_KEY configured in environment or .env file."
            }

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": "Hello Gemini, confirm status."}]}]
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
            except Exception as e:
                logger.warning(f"Gemini API test error ({model}): {e}")

        return {
            "success": False,
            "status": "failed",
            "message": "Unable to connect to Google Gemini API."
        }

    def summarize(self, blocks: List[Dict[str, Any]]) -> Optional[str]:
        """Generate a concise clinical summary of document text blocks using Gemini API."""
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key or GeminiQAClient._KEY_INVALID:
            return None

        formatted_blocks = []
        for idx, b in enumerate(blocks, 1):
            page_num = b.get("page_number", 1)
            text = b.get("text", "").strip()
            formatted_blocks.append(f"[Chunk {idx}] (Page {page_num}):\n{text}")

        blocks_str = "\n\n".join(formatted_blocks)

        prompt = f"""You are a Retrieval-Augmented Generation (RAG) medical report engine.
Summarize the following retrieved document chunks into a concise 2-sentence clinical summary.

Retrieved Document Chunks:
{blocks_str}

Summary:"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
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
                                return summary_text
            except Exception as e:
                logger.warning(f"Error calling Gemini API for summary ({model}): {e}")
                GeminiQAClient._KEY_INVALID = True
                break

        return None

    def query(self, question: str, top_chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Send Top 5 retrieved document chunks and question to Gemini RAG engine.
        Returns structured JSON with matched_text, page, answer, reason, and confidence.
        """
        api_key = self.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key or GeminiQAClient._KEY_INVALID:
            return None

        if not top_chunks:
            logger.info("No chunks provided to Gemini RAG client.")
            return None

        formatted_chunks = []
        for idx, c in enumerate(top_chunks, 1):
            page_num = c.get("page_number", 1)
            text = c.get("text", "").strip()
            formatted_chunks.append(f"[Retrieved Chunk {idx}] (Page {page_num}):\n{text}")

        chunks_str = "\n\n".join(formatted_chunks)

        prompt = f"""You are a Retrieval-Augmented Generation (RAG) system for medical document analysis.
Your job is to answer the user's question using ONLY the retrieved document chunks provided below.

CRITICAL RULES:
1. Base your answer STRICTLY on the retrieved document chunks.
2. DO NOT use external medical knowledge.
3. DO NOT hallucinate or infer facts not explicitly in the text.
4. If the requested information is NOT in the retrieved chunks, return JSON:
   {{"answer": "The uploaded report does not contain this information.", "matched_text": "", "page": null, "reason": "Information not found in retrieved chunks", "confidence": 0.0}}
5. If found, return STRICT JSON with these exact 5 keys:
   - "answer": concise answer string (e.g. "MANJIT SINGH", "1.02 mg/dL", "Normal Sinus Rhythm")
   - "matched_text": the exact line or phrase from the retrieved chunk containing the answer (e.g. "Patient Name: MANJIT SINGH" or "Serum Creatinine: 1.02 mg/dL")
   - "page": integer page number where the evidence is located
   - "reason": brief 1-sentence technical justification for the answer
   - "confidence": float score between 0.0 and 1.0 (e.g. 0.98)
6. Respond ONLY with valid JSON. No markdown backticks, no markdown formatting outside JSON.

Retrieved Document Chunks:
{chunks_str}

User Question: {question}
"""

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.9,
                "maxOutputTokens": 300,
                "responseMimeType": "application/json"
            }
        }

        for model in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=2)
                if response.status_code == 200:
                    res_data = response.json()
                    candidates = res_data.get("candidates", [])
                    if candidates:
                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        if content_parts:
                            raw_text = content_parts[0].get("text", "").strip()
                            parsed_json = self._parse_json_response(raw_text)
                            if parsed_json:
                                logger.info(f"Gemini RAG ({model}) returned answer: {parsed_json.get('answer')}")
                                return parsed_json
                elif response.status_code in (400, 401, 403):
                    GeminiQAClient._KEY_INVALID = True
                    break
            except Exception as e:
                logger.warning(f"Error calling Gemini RAG ({model}): {e}")
                GeminiQAClient._KEY_INVALID = True
                break

        logger.warning("Gemini API models unavailable or non-responsive. Using local RAG fallback engine.")
        return None

    @staticmethod
    def _parse_json_response(raw_text: str) -> Optional[Dict[str, Any]]:
        """Parse raw text response from Gemini into a structured JSON dict."""
        if not raw_text:
            return None

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()

        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                answer = str(data.get("answer", "")).strip()
                matched_text = str(data.get("matched_text", "")).strip()
                reason = str(data.get("reason", "")).strip()
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
                    "answer": answer,
                    "matched_text": matched_text,
                    "page": page,
                    "reason": reason,
                    "confidence": round(confidence, 2)
                }
        except Exception as e:
            logger.warning(f"Failed to parse Gemini RAG JSON output: {e}. Raw: '{raw_text}'")

        return None
