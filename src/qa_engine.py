"""Document-grounded question answering for uploaded PDFs.

The engine turns the visible text of the active PDF into generic block/value
records, retrieves the most relevant record for a user question, and returns
that value together with a crop of the matching document block.
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
    "give", "i", "in", "is", "me", "of", "on", "please", "show", "tell",
    "the", "this", "to", "value", "what", "where", "which", "who", "with",
})
_SUMMARY_WORDS = frozenset({"summary", "summarise", "summarize", "overview", "brief"})


@dataclass
class FieldRecord:
    field_name: str
    field_value: str
    full_line_text: str
    page_number: int
    bounding_box: List[float]


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
    field_records: List[FieldRecord]
    record_search_texts: List[str]
    vectorizer: Optional[TfidfVectorizer]
    field_embeddings: Optional[np.ndarray]


class DocumentQAEngine:
    """Answers questions using only text and geometry extracted from one PDF."""

    def __init__(self, outputs_dir: Path = OUTPUTS_DIR):
        self.outputs_dir = Path(outputs_dir)
        self.snippets_dir = self.outputs_dir / "qa_snippets"
        self.snippets_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[DocumentSession] = None

    def purge_and_create_session(self, pdf_path: str | Path, document_name: Optional[str] = None) -> str:
        """Discard the old document and build a fresh, isolated index."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self._purge_snippets()
        self.current_session = None
        session_id = str(uuid.uuid4())
        pages: List[Dict[str, Any]] = []
        records: List[FieldRecord] = []

        with fitz.open(pdf_path) as document:
            for page_index, page in enumerate(document):
                page_number = page_index + 1
                raw_text = page.get_text("text")
                page_blocks = self._extract_page_blocks(page, page_number)
                page_records = self._records_from_blocks(page_blocks, page_number)
                records.extend(page_records)
                pages.append({
                    "page_number": page_number,
                    "raw_text": raw_text,
                    "clean_text": raw_text.casefold(),
                    "blocks": [{"text": text, "bbox": bbox} for text, bbox in page_blocks],
                    "rect": (page.rect.width, page.rect.height),
                })

        records = self._deduplicate_records(records)
        search_texts = [self._search_text(record) for record in records]
        vectorizer: Optional[TfidfVectorizer] = None
        embeddings: Optional[np.ndarray] = None
        if search_texts:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            embeddings = vectorizer.fit_transform(search_texts).toarray()

        self.current_session = DocumentSession(
            session_id=session_id,
            document_name=document_name or pdf_path.name,
            pdf_path=pdf_path,
            indexed_pages=pages,
            field_records=records,
            record_search_texts=search_texts,
            vectorizer=vectorizer,
            field_embeddings=embeddings,
        )
        logger.info("Indexed %s records from %s", len(records), self.current_session.document_name)
        return session_id

    def ask(self, question: str, session_id: Optional[str] = None) -> QAResult:
        if not self.current_session:
            return self._build_not_found_result(question)
        session = self.current_session
        if session_id and session_id != session.session_id:
            raise ValueError("The supplied session ID does not match the active document.")

        normalized_question = self._normalise(question)
        if not normalized_question:
            return self._build_not_found_result(question)
        if self._is_summary_query(normalized_question):
            return self._build_summary_result(question)
        if not session.vectorizer or session.field_embeddings is None:
            return self._build_not_found_result(question)

        query = self._retrieval_query(normalized_question)
        if not query:
            return self._build_not_found_result(question)

        scores = cosine_similarity(session.vectorizer.transform([query]), session.field_embeddings)[0]
        top_index = int(np.argmax(scores))
        score = float(scores[top_index])
        if score < 0.08:
            return self._build_not_found_result(question)

        record = session.field_records[top_index]
        value = self._clean_value(record.field_value)
        if not value:
            return self._build_not_found_result(question)

        confidence = round(min(0.99, 0.35 + score * 1.5), 2)
        snippet_name = f"crop_session_{session.session_id[:8]}_p{record.page_number}_{uuid.uuid4().hex[:8]}.png"
        return self._build_qa_result(
            question=question,
            answer=value,
            field=record.field_name,
            value=value,
            page_num=record.page_number,
            sec_page_num=None,
            confidence=confidence,
            section_title=record.field_name or "Document text",
            crop_bbox=record.bounding_box,
            snippet_filename=snippet_name,
        )

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.casefold().split())

    def _extract_page_blocks(self, page: fitz.Page, page_number: int) -> List[Tuple[str, List[float]]]:
        """Return text blocks with normalized rectangles in reading order."""
        blocks: List[Tuple[str, List[float]]] = []
        page_dict = page.get_text("dict")
        width, height = page.rect.width, page.rect.height
        for block in page_dict.get("blocks", []):
            text_lines: List[str] = []
            block_bbox: Optional[List[float]] = None
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                bbox = line.get("bbox")
                if text and bbox and width and height:
                    norm_bbox = self._normalised_bbox(bbox, width, height)
                    text_lines.append(text)
                    block_bbox = self._merge_bbox(norm_bbox, block_bbox or norm_bbox) if block_bbox else norm_bbox
            block_text = self._clean_value("\n".join(text_lines))
            if block_text and block_bbox:
                blocks.append((block_text, block_bbox))
        return blocks

    def _records_from_blocks(self, blocks: List[Tuple[str, List[float]]], page_number: int) -> List[FieldRecord]:
        """Build records from generic document blocks and their layout."""
        records: List[FieldRecord] = []
        for text, bbox in blocks:
            parsed = self._record_from_block(text, bbox, page_number)
            if parsed is not None:
                records.append(parsed)
        return records

    def _record_from_block(self, text: str, bbox: List[float], page_number: int) -> Optional[FieldRecord]:
        cleaned = self._clean_value(text)
        if not cleaned:
            return None

        split_pair = self._split_inline_pair(cleaned)
        if split_pair is not None:
            label, value = split_pair
            return FieldRecord(label, value, cleaned, page_number, bbox)

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) > 1:
            first_line = lines[0]
            rest_text = self._clean_value("\n".join(lines[1:]))
            if self._looks_like_label(first_line) and self._looks_like_value(rest_text):
                return FieldRecord(first_line, rest_text, cleaned, page_number, bbox)

        if len(lines) > 1 and self._looks_like_value(cleaned):
            return FieldRecord("Document block", cleaned, cleaned, page_number, bbox)

        return FieldRecord("Document block", cleaned, cleaned, page_number, bbox)

    @staticmethod
    def _same_visual_row(first: List[float], second: List[float]) -> bool:
        first_center = (first[1] + first[3]) / 2
        second_center = (second[1] + second[3]) / 2
        return abs(first_center - second_center) <= 0.012

    @staticmethod
    def _split_inline_pair(text: str) -> Optional[Tuple[str, str]]:
        match = re.match(r"^\s*(.+?)\s*(?::|\t+|\.{3,}|\s[-\u2013\u2014]\s)\s*(.+?)\s*$", text)
        if not match:
            return None
        label, value = (part.strip() for part in match.groups())
        if not label or not value or len(label) > 120:
            return None
        return label, value

    @staticmethod
    def _looks_like_label(text: str) -> bool:
        words = re.findall(r"[\w']+", text)
        return bool(words) and len(text) <= 80 and len(words) <= 10 and not text.rstrip().endswith((".", "?", "!"))

    @staticmethod
    def _looks_like_value(text: str) -> bool:
        return bool(text.strip()) and len(text) <= 240 and not text.rstrip().endswith("?")

    @staticmethod
    def _normalised_bbox(bbox: Tuple[float, float, float, float], width: float, height: float) -> List[float]:
        return [max(0.0, min(1.0, bbox[0] / width)), max(0.0, min(1.0, bbox[1] / height)),
                max(0.0, min(1.0, bbox[2] / width)), max(0.0, min(1.0, bbox[3] / height))]

    @staticmethod
    def _merge_bbox(first: List[float], second: List[float]) -> List[float]:
        return [min(first[0], second[0]), min(first[1], second[1]), max(first[2], second[2]), max(first[3], second[3])]

    @staticmethod
    def _deduplicate_records(records: List[FieldRecord]) -> List[FieldRecord]:
        unique: List[FieldRecord] = []
        seen = set()
        for record in records:
            key = (record.page_number, record.field_name.casefold(), record.field_value.casefold(), tuple(round(v, 4) for v in record.bounding_box))
            if key not in seen:
                seen.add(key)
                unique.append(record)
        return unique

    @staticmethod
    def _search_text(record: FieldRecord) -> str:
        return f"{record.field_name} {record.field_value} {record.full_line_text}"

    @staticmethod
    def _retrieval_query(question: str) -> str:
        tokens = re.findall(r"[\w']+", question)
        return " ".join(token for token in tokens if token not in _STOP_WORDS)

    @staticmethod
    def _clean_value(value: str) -> str:
        return " ".join(value.replace("\u00a0", " ").split()).strip(" :-\u2013\u2014")

    @staticmethod
    def _is_summary_query(question: str) -> bool:
        return bool(_SUMMARY_WORDS.intersection(re.findall(r"[\w']+", question)))

    def _build_summary_result(self, question: str) -> QAResult:
        assert self.current_session is not None
        session = self.current_session
        excerpts: List[str] = []
        seen = set()
        for page in session.indexed_pages:
            for block in page["blocks"]:
                text = self._clean_value(block["text"])
                key = text.casefold()
                if len(text) >= 12 and key not in seen and self._looks_like_value(text):
                    excerpts.append(text)
                    seen.add(key)
                if len(excerpts) == 3:
                    break
            if len(excerpts) == 3:
                break
        if not excerpts:
            return self._build_not_found_result(question)
        answer = "Summary from the uploaded PDF: " + " | ".join(excerpts)
        first_block = next((block for page in session.indexed_pages for block in page["blocks"] if block["text"]), None)
        bbox = first_block["bbox"] if first_block else None
        return self._build_qa_result(
            question=question,
            answer=answer,
            field="Document summary",
            value=answer,
            page_num=1 if bbox else None,
            sec_page_num=None,
            confidence=0.8,
            section_title="Document-derived summary",
            crop_bbox=bbox,
            snippet_filename=f"crop_session_{session.session_id[:8]}_summary.png" if bbox else None,
        )

    def _build_not_found_result(self, question: str) -> QAResult:
        session = self.current_session
        return QAResult(question, "The uploaded PDF does not contain this information.", None, None,
                        None, None, 0.0, "No matching document evidence", None, None, None,
                        session.session_id if session else "none", session.document_name if session else "none")

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
        return QAResult(question, answer, field, value, page_num, sec_page_num, confidence, section_title,
                        crop_bbox, snippet_filename, str(snippet_path) if snippet_path and snippet_path.exists() else None,
                        self.current_session.session_id, self.current_session.document_name)

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
        except Exception:
            logger.exception("Unable to create QA evidence crop")

    def _purge_snippets(self) -> None:
        for snippet in self.snippets_dir.glob("*.png"):
            try:
                snippet.unlink()
            except OSError:
                logger.warning("Could not remove old snippet %s", snippet)

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Return document-neutral examples; the uploaded document supplies the answer."""
        return [
            {"icon": "🔎", "question": "What is the document number?", "tag": "Lookup", "page": 1},
            {"icon": "👤", "question": "Who is the named person?", "tag": "Details", "page": 1},
            {"icon": "📅", "question": "What is the date?", "tag": "Details", "page": 1},
            {"icon": "📋", "question": "Summarize this document.", "tag": "Summary", "page": 1},
        ]
