"""Generic document-grounded question answering for uploaded PDFs.

The engine indexes all searchable text blocks in the uploaded PDF, retrieves the
most relevant block for each question, and extracts the answer value from the
matched block. It does not depend on document-specific vocabulary or field
layouts.
"""

import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import OUTPUTS_DIR

logger = logging.getLogger("qa_engine")

_STOP_WORDS = frozenset({
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "from",
    "i", "in", "is", "me", "of", "on", "please", "show", "tell",
    "the", "this", "to", "value", "what", "where", "which", "who", "with",
})


@dataclass
class QAResult:
    question: str
    answer: str
    field: Optional[str]
    value: Optional[str]
    page_number: Optional[int]
    secondary_page_number: Optional[int]
    confidence: float
    section_title: str
    bounding_box: Optional[List[float]]
    snippet_filename: Optional[str]
    snippet_path: Optional[str]
    session_id: str
    document_name: str


@dataclass
class DocumentSession:
    session_id: str
    document_name: str
    pdf_path: Path
    indexed_pages: List[Dict[str, Any]]
    indexed_blocks: List[Dict[str, Any]]
    block_search_texts: List[str]
    vectorizer: Optional[TfidfVectorizer]
    block_embeddings: Optional[np.ndarray]


class DocumentQAEngine:
    """Answers questions using only text and geometry extracted from one PDF."""

    def __init__(self, outputs_dir: Path = OUTPUTS_DIR):
        self.outputs_dir = Path(outputs_dir)
        self.snippets_dir = self.outputs_dir / "qa_snippets"
        self.snippets_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[DocumentSession] = None

    def purge_and_create_session(self, pdf_path: str | Path, document_name: Optional[str] = None) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            logger.error("PDF file not found: %s", pdf_path)
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self._purge_snippets()
        self.current_session = None
        session_id = str(uuid.uuid4())
        indexed_pages: List[Dict[str, Any]] = []
        indexed_blocks: List[Dict[str, Any]] = []

        logger.info("Creating QA session %s for document %s", session_id[:8], pdf_path)
        try:
            with fitz.open(pdf_path) as document:
                for page_index, page in enumerate(document):
                    page_number = page_index + 1
                    logger.debug("Extracting blocks from page %s", page_number)
                    blocks = self._extract_blocks(page, page_number)
                    indexed_blocks.extend(blocks)
                    try:
                        raw_text = page.get_text("text")
                    except Exception:
                        logger.exception("Error extracting raw text from page %s", page_number)
                        raw_text = ""
                    indexed_pages.append({
                        "page_number": page_number,
                        "raw_text": raw_text,
                        "blocks": blocks,
                    })

            block_search_texts = [block["normalized_text"] for block in indexed_blocks]
            vectorizer: Optional[TfidfVectorizer] = None
            embeddings: Optional[np.ndarray] = None
            if block_search_texts:
                vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
                embeddings = vectorizer.fit_transform(block_search_texts).toarray()

            self.current_session = DocumentSession(
                session_id=session_id,
                document_name=document_name or pdf_path.name,
                pdf_path=pdf_path,
                indexed_pages=indexed_pages,
                indexed_blocks=indexed_blocks,
                block_search_texts=block_search_texts,
                vectorizer=vectorizer,
                block_embeddings=embeddings,
            )

            logger.info("Indexed %s blocks from %s", len(indexed_blocks), self.current_session.document_name)
            return session_id

        except Exception:
            logger.exception("Failed to create QA session for %s", pdf_path)
            raise

    def ask(self, question: str, session_id: Optional[str] = None) -> QAResult:
        logger.info("QA ask invoked: '%s' (session=%s)", question, session_id or (self.current_session.session_id if self.current_session else 'none'))
        if not self.current_session:
            logger.warning("No active QA session when asking question: %s", question)
            return self._build_not_found_result(question)
        session = self.current_session
        if session_id and session_id != session.session_id:
            raise ValueError("The supplied session ID does not match the active document.")

        normalized_question = self._normalise(question)
        if not normalized_question:
            return self._build_not_found_result(question)

        if any(keyword in normalized_question for keyword in ("summarize", "summary")):
            return self._build_summary_result(question)

        if not session.vectorizer or session.block_embeddings is None:
            return self._build_not_found_result(question)

        query = self._retrieval_query(normalized_question)
        if not query:
            return self._build_not_found_result(question)

        query_tokens = set(self._tokenize(normalized_question))
        scores = cosine_similarity(session.vectorizer.transform([query]), session.block_embeddings)[0]
        ranked_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)

        top_index = None
        top_score = 0.0
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < 0.10:
                break
            block = session.indexed_blocks[idx]
            if self._is_relevant_block(block, query_tokens, score):
                top_index = idx
                top_score = score
                break

        if top_index is None:
            logger.info("No relevant block found for question: %s", question)
            return self._build_not_found_result(question)

        block = session.indexed_blocks[top_index]
        answer_value = self._extract_answer_from_block(block, normalized_question)
        if not answer_value:
            answer_value = self._clean_value(block["text"])
        if not answer_value:
            logger.info("Extracted empty answer for question: %s after inspecting block on page %s", question, block.get('page_number'))
            return self._build_not_found_result(question)

        confidence = round(min(0.99, 0.25 + top_score * 0.75), 2)
        snippet_name = f"crop_session_{session.session_id[:8]}_p{block['page_number']}_{uuid.uuid4().hex[:8]}.png"
        return self._build_qa_result(
            question=question,
            answer=answer_value,
            field="Document block",
            value=answer_value,
            page_num=block["page_number"],
            sec_page_num=None,
            confidence=confidence,
            section_title=self._short_title(block["text"]),
            crop_bbox=block["bbox"],
            snippet_filename=snippet_name,
        )

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [
            token
            for token in re.findall(r"[\w']+", text)
            if token and token not in _STOP_WORDS
        ]

    @staticmethod
    def _retrieval_query(normalized_question: str) -> str:
        tokens = DocumentQAEngine._tokenize(normalized_question)
        return " ".join(tokens)

    def _extract_blocks(self, page: fitz.Page, page_number: int) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        page_dict = page.get_text("dict")
        width, height = page.rect.width, page.rect.height
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            text_lines: List[str] = []
            normalized_bbox: Optional[List[float]] = None
            for line in block.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not line_text:
                    continue
                bbox = line.get("bbox")
                if bbox and width and height:
                    line_bbox = self._normalised_bbox(bbox, width, height)
                    normalized_bbox = self._merge_bbox(normalized_bbox, line_bbox) if normalized_bbox else line_bbox
                text_lines.append(line_text)
            if not text_lines or not normalized_bbox:
                continue
            text = "\n".join(text_lines)
            normalized_text = self._normalise(text)
            blocks.append({
                "page_number": page_number,
                "bbox": normalized_bbox,
                "text": text,
                "normalized_text": normalized_text,
                "tokens": self._tokenize(normalized_text),
            })
        return blocks

    def _extract_answer_from_block(self, block: Dict[str, Any], normalized_question: str) -> str:
        lines = [line.strip() for line in block["text"].splitlines() if line.strip()]
        pairs = self._extract_label_value_pairs(lines)
        question_tokens = set(self._tokenize(normalized_question))
        if pairs:
            best_pair = max(pairs, key=lambda pair: self._pair_score(pair, question_tokens))
            if self._pair_score(best_pair, question_tokens) > 0:
                return self._clean_value(best_pair["value"])
            if len(pairs) == 1:
                return self._clean_value(pairs[0]["value"])

        if len(lines) >= 2 and self._looks_like_label(lines[0]) and self._looks_like_value(lines[1]):
            return self._clean_value(lines[1])

        if len(lines) == 1:
            single = lines[0]
            pair = self._split_label_value_line(single)
            if pair is not None:
                return self._clean_value(pair[1])
            return self._clean_value(single)

        if lines:
            longest_line = max(lines, key=len)
            return self._clean_value(longest_line)

        return ""

    def _extract_label_value_pairs(self, lines: List[str]) -> List[Dict[str, Any]]:
        pairs: List[Dict[str, Any]] = []
        for line in lines:
            split = self._split_label_value_line(line)
            if split is not None:
                label, value = split
                pairs.append({
                    "label": label,
                    "value": value,
                    "tokens": set(self._tokenize(self._normalise(label))),
                })
        return pairs

    @staticmethod
    def _split_label_value_line(line: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"^\s*(.+?)\s*(?:[:\t\-–—]|\.\.\.+|\s{2,})\s*(.+?)\s*$", line)
        if not match:
            return None
        label, value = match.groups()
        label = label.strip()
        value = value.strip()
        if not label or not value or len(label) > 120:
            return None
        return label, value

    @staticmethod
    def _pair_score(pair: Dict[str, Any], query_tokens: set) -> float:
        label_overlap = len(pair["tokens"].intersection(query_tokens))
        value_tokens = set(re.findall(r"[\w']+", pair["value"].casefold()))
        value_overlap = len(value_tokens.intersection(query_tokens))
        return float(label_overlap) + 0.25 * float(value_overlap)

    @staticmethod
    def _looks_like_label(text: str) -> bool:
        words = re.findall(r"[\w']+", text)
        return bool(words) and len(words) <= 8 and len(text) <= 120 and not text.endswith("?")

    @staticmethod
    def _looks_like_value(text: str) -> bool:
        return bool(text.strip()) and len(text) <= 240 and not text.endswith("?")

    @staticmethod
    def _short_title(text: str) -> str:
        title = text.strip().splitlines()[0]
        return title if len(title) <= 60 else title[:57] + "..."

    def _is_relevant_block(self, block: Dict[str, Any], query_tokens: set, score: float) -> bool:
        if score >= 0.35:
            return True
        block_tokens = set(block.get("tokens", []))
        overlap = len(block_tokens.intersection(query_tokens))
        return overlap >= 2 and score >= 0.15

    @staticmethod
    def _normalised_bbox(bbox: Tuple[float, float, float, float], width: float, height: float) -> List[float]:
        return [
            max(0.0, min(1.0, bbox[0] / width)),
            max(0.0, min(1.0, bbox[1] / height)),
            max(0.0, min(1.0, bbox[2] / width)),
            max(0.0, min(1.0, bbox[3] / height)),
        ]

    @staticmethod
    def _merge_bbox(first: Optional[List[float]], second: List[float]) -> List[float]:
        if first is None:
            return second
        return [
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        ]

    def _build_summary_result(self, question: str) -> QAResult:
        assert self.current_session is not None
        session = self.current_session
        excerpts: List[str] = []
        seen = set()
        for page in session.indexed_pages:
            for block in page["blocks"]:
                text = self._clean_value(block["text"])
                key = text.casefold()
                if key not in seen and len(text) >= 40:
                    excerpts.append(text)
                    seen.add(key)
                if len(excerpts) == 3:
                    break
            if len(excerpts) == 3:
                break
        if not excerpts:
            return self._build_not_found_result(question)
        answer = " ".join(excerpts)
        first_block = next((block for page in session.indexed_pages for block in page["blocks"] if block["text"]), None)
        bbox = first_block["bbox"] if first_block else None
        return self._build_qa_result(
            question=question,
            answer=answer,
            field="Document summary",
            value=answer,
            page_num=first_block["page_number"] if first_block else None,
            sec_page_num=None,
            confidence=0.80,
            section_title="Document summary",
            crop_bbox=bbox,
            snippet_filename=f"crop_session_{session.session_id[:8]}_summary.png" if bbox else None,
        )

    @staticmethod
    def _clean_value(value: str) -> str:
        return " ".join(value.replace("\u00a0", " ").split()).strip(" :-–—")

    def _build_not_found_result(self, question: str) -> QAResult:
        session = self.current_session
        return QAResult(
            question,
            "The uploaded PDF does not contain this information.",
            None,
            None,
            None,
            None,
            0.0,
            "No matching document evidence",
            None,
            None,
            None,
            session.session_id if session else "none",
            session.document_name if session else "none",
        )

    def _build_qa_result(self, question: str, answer: str, field: Optional[str], value: Optional[str],
                         page_num: Optional[int], sec_page_num: Optional[int], confidence: float,
                         section_title: str, crop_bbox: Optional[List[float]], snippet_filename: Optional[str]) -> QAResult:
        assert self.current_session is not None
        snippet_path = None
        if self._valid_bbox(crop_bbox) and page_num is not None and snippet_filename:
            snippet_path = self.snippets_dir / snippet_filename
            self._crop_snippet_from_pdf(self.current_session.pdf_path, page_num, crop_bbox, snippet_path)
        else:
            crop_bbox, snippet_filename = None, None
        return QAResult(
            question,
            answer,
            field,
            value,
            page_num,
            sec_page_num,
            confidence,
            section_title,
            crop_bbox,
            snippet_filename,
            str(snippet_path) if snippet_path and snippet_path.exists() else None,
            self.current_session.session_id,
            self.current_session.document_name,
        )

    @staticmethod
    def _valid_bbox(bbox: Optional[List[float]]) -> bool:
        return bool(bbox and len(bbox) == 4 and 0 <= bbox[0] < bbox[2] <= 1 and 0 <= bbox[1] < bbox[3] <= 1)

    def _crop_snippet_from_pdf(self, pdf_path: Path, page_num: int, bbox: List[float], output_path: Path) -> None:
        try:
            with fitz.open(pdf_path) as document:
                if not 1 <= page_num <= len(document):
                    return
                pix = document[page_num - 1].get_pixmap(dpi=250)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            width, height = image.size
            x1, y1, x2, y2 = int(bbox[0] * width), int(bbox[1] * height), int(bbox[2] * width), int(bbox[3] * height)
            padding = 12
            cropped = image.crop((max(0, x1 - padding), max(0, y1 - padding), min(width, x2 + padding), min(height, y2 + padding)))
            ImageDraw.Draw(cropped).rectangle([(2, 2), (cropped.width - 3, cropped.height - 3)], outline="#10b981", width=6)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path, format="PNG")
            logger.info("Saved QA evidence crop to %s", output_path)
        except Exception:
            logger.exception("Unable to create QA evidence crop")

    def _purge_snippets(self) -> None:
        for snippet in self.snippets_dir.glob("*.png"):
            try:
                snippet.unlink()
            except OSError:
                logger.warning("Could not remove old snippet %s", snippet)

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        return [
            {"icon": "🔎", "question": "What is the document number?", "tag": "Lookup", "page": 1},
            {"icon": "👤", "question": "Who is the named person?", "tag": "Details", "page": 1},
            {"icon": "📅", "question": "What is the date?", "tag": "Details", "page": 1},
            {"icon": "📋", "question": "Summarize this document.", "tag": "Summary", "page": 1},
        ]
