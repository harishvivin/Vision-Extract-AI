"""Document-grounded question answering for uploaded PDFs.

The engine deliberately has no document-type vocabulary.  It turns the visual
text of the active PDF into generic field/value records, retrieves the most
relevant record, and returns evidence cropped from that record's real location.
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
# These are document-role terms, not medical test names.  Expanding a natural
# language role lets a question such as "patient name" find the equivalent
# label used by a particular issuer (for example, applicant or proposer).
_PERSON_ROLE_TERMS = frozenset({
    "applicant", "client", "customer", "employee", "examinee", "insured",
    "member", "patient", "policyholder", "proposer", "subject",
})


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
                lines = self._extract_lines(page)
                raw_text = page.get_text("text")
                page_records = self._records_from_lines(lines, page_number)
                records.extend(page_records)
                pages.append({
                    "page_number": page_number,
                    "raw_text": raw_text,
                    "clean_text": raw_text.casefold(),
                    "blocks": [{"text": text, "bbox": bbox} for text, bbox in lines],
                    "rect": (page.rect.width, page.rect.height),
                })

        records = self._deduplicate_records(records)
        search_texts = [self._search_text(record) for record in records]
        vectorizer: Optional[TfidfVectorizer] = None
        embeddings: Optional[np.ndarray] = None
        if search_texts:
            # Character n-grams make retrieval tolerant of punctuation, OCR spacing,
            # spelling variants, and field names expressed with different separators.
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
        # When structured records exist, a standalone heading must not beat its
        # accompanying value merely because its label is a shorter string.
        structured = np.array([
            self._normalise(record.field_name) != self._normalise(record.field_value)
            for record in session.field_records
        ])
        if structured.any():
            scores = np.where(structured, scores, -1.0)
        top_index = int(np.argmax(scores))
        score = float(scores[top_index])
        # Character TF-IDF scores are deliberately unmodified.  A modest floor stops
        # unrelated questions being answered merely because they share common letters.
        if score < 0.11:
            return self._build_not_found_result(question)

        record = session.field_records[top_index]
        value = self._clean_value(record.field_value)
        if not value:
            return self._build_not_found_result(question)
        confidence = round(min(0.99, 0.45 + score * 1.5), 2)
        snippet_name = f"crop_session_{session.session_id[:8]}_p{record.page_number}_{uuid.uuid4().hex[:8]}.png"
        return self._build_qa_result(
            question=question,
            answer=f"{value} (Page {record.page_number})",
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

    def _extract_lines(self, page: fitz.Page) -> List[Tuple[str, List[float]]]:
        """Return visual PDF lines and their normalized rectangles in reading order."""
        lines: List[Tuple[str, List[float]]] = []
        page_dict = page.get_text("dict")
        width, height = page.rect.width, page.rect.height
        for block in page_dict.get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                bbox = line.get("bbox")
                if text and bbox and width and height:
                    lines.append((text, self._normalised_bbox(bbox, width, height)))
        return lines

    def _records_from_lines(self, lines: List[Tuple[str, List[float]]], page_number: int) -> List[FieldRecord]:
        """Build records from inline text, visual rows, and finally vertical forms.

        PDF generators often place a label, separator, and value in separate text
        fragments.  The fragments may even be returned out of reading order, so
        adjacent-list matching is insufficient.  Separator anchors let us rebuild
        each visual row without relying on a report-specific field name.
        """
        records: List[FieldRecord] = []
        consumed = set()

        # Prefer explicit label : value rows.  This produces one evidence rectangle
        # around both pieces and prevents a bare label winning retrieval.
        for separator_index, (separator_text, separator_bbox) in enumerate(lines):
            separator = separator_text.strip()
            if not separator.startswith(":"):
                continue
            label_index = self._nearest_left_fragment(lines, separator_index)
            if label_index is None:
                continue
            label, label_bbox = lines[label_index]
            inline_value = separator[1:].strip()
            if inline_value:
                value, value_bbox = inline_value, separator_bbox
                value_index = separator_index
            else:
                value_index = self._nearest_right_fragment(lines, separator_index)
                if value_index is None:
                    continue
                value, value_bbox = lines[value_index]
            continuation_indices = self._value_continuations(lines, value_index, value_bbox)
            continuation_texts = [lines[index][0] for index in continuation_indices]
            evidence_bbox = value_bbox
            for continuation_index in continuation_indices:
                evidence_bbox = self._merge_bbox(evidence_bbox, lines[continuation_index][1])
            value = self._clean_value(" ".join([value, *continuation_texts]))
            if not value:
                continue
            records.append(FieldRecord(
                self._clean_value(label), value, f"{label}: {value}", page_number,
                self._merge_bbox(label_bbox, evidence_bbox),
            ))
            consumed.update({label_index, separator_index, value_index, *continuation_indices})

        for index, (text, bbox) in enumerate(lines):
            if index in consumed:
                continue
            inline = self._split_inline_pair(text)
            if inline:
                label, value = inline
                records.append(FieldRecord(label, value, text, page_number, bbox))
                continue
            next_line = lines[index + 1] if index + 1 < len(lines) else None
            # Consecutive visual lines are a form label/value pair only when they are
            # close, aligned, and the first line looks label-like.  No domain terms are used.
            if next_line:
                next_text, next_bbox = next_line
            if next_line and self._looks_like_label(text) and self._looks_like_value(next_text) and self._are_nearby(bbox, next_bbox):
                records.append(FieldRecord(
                    text, next_text, f"{text}: {next_text}", page_number,
                    self._merge_bbox(bbox, next_bbox),
                ))
            else:
                # Include standalone text so paragraph questions remain answerable.
                records.append(FieldRecord(text, text, text, page_number, bbox))
        return records

    @staticmethod
    def _same_visual_row(first: List[float], second: List[float]) -> bool:
        first_center = (first[1] + first[3]) / 2
        second_center = (second[1] + second[3]) / 2
        return abs(first_center - second_center) <= 0.012

    def _nearest_left_fragment(self, lines: List[Tuple[str, List[float]]], anchor_index: int) -> Optional[int]:
        anchor_bbox = lines[anchor_index][1]
        candidates = [
            index for index, (text, bbox) in enumerate(lines)
            if index != anchor_index and text.strip() and not text.strip().startswith(":")
            and bbox[2] <= anchor_bbox[0] + 0.002 and self._same_visual_row(bbox, anchor_bbox)
        ]
        return max(candidates, key=lambda index: lines[index][1][2], default=None)

    def _nearest_right_fragment(self, lines: List[Tuple[str, List[float]]], anchor_index: int) -> Optional[int]:
        anchor_bbox = lines[anchor_index][1]
        candidates = [
            index for index, (text, bbox) in enumerate(lines)
            if index != anchor_index and text.strip() and not text.strip().startswith(":")
            and bbox[0] >= anchor_bbox[2] - 0.002 and self._same_visual_row(bbox, anchor_bbox)
        ]
        return min(candidates, key=lambda index: lines[index][1][0], default=None)

    def _value_continuations(self, lines: List[Tuple[str, List[float]]], value_index: int,
                             value_bbox: List[float]) -> List[int]:
        """Return wrapped value fragments directly beneath the first value fragment."""
        continuations: List[int] = []
        current_bbox = value_bbox
        while True:
            candidates = [
                index for index, (text, bbox) in enumerate(lines)
                if index != value_index and index not in continuations and text.strip()
                and not text.strip().startswith(":")
                and current_bbox[3] - 0.002 <= bbox[1] <= current_bbox[3] + 0.03
                and abs(bbox[0] - value_bbox[0]) <= 0.035
                and not self._has_separator_to_left(lines, index)
            ]
            if not candidates:
                return continuations
            next_index = min(candidates, key=lambda index: lines[index][1][1])
            continuations.append(next_index)
            current_bbox = lines[next_index][1]

    def _has_separator_to_left(self, lines: List[Tuple[str, List[float]]], value_index: int) -> bool:
        value_bbox = lines[value_index][1]
        return any(
            text.strip().startswith(":") and bbox[2] <= value_bbox[0] + 0.002
            and self._same_visual_row(bbox, value_bbox)
            for index, (text, bbox) in enumerate(lines) if index != value_index
        )

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
        return bool(text.strip()) and len(text) <= 160 and not text.rstrip().endswith("?")

    @staticmethod
    def _are_nearby(first: List[float], second: List[float]) -> bool:
        vertical_gap = second[1] - first[3]
        horizontal_offset = abs(first[0] - second[0])
        return -0.02 <= vertical_gap <= 0.055 and horizontal_offset <= 0.16

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
        # Repeating the label gives it appropriate prominence without query-specific boosts.
        return f"{record.field_name} {record.field_name} {record.field_value} {record.full_line_text}"

    @staticmethod
    def _retrieval_query(question: str) -> str:
        tokens = re.findall(r"[\w']+", question)
        meaningful = [token for token in tokens if token not in _STOP_WORDS]
        if _PERSON_ROLE_TERMS.intersection(meaningful):
            meaningful.extend(sorted(_PERSON_ROLE_TERMS))
        return " ".join(meaningful)

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
                if len(text) >= 20 and key not in seen:
                    excerpts.append(text)
                    seen.add(key)
                if len(excerpts) == 3:
                    break
            if len(excerpts) == 3:
                break
        if not excerpts:
            return self._build_not_found_result(question)
        answer = f"Summary of {session.document_name} ({len(session.indexed_pages)} pages): " + " ".join(excerpts)
        first_page = session.indexed_pages[0]
        bbox = first_page["blocks"][0]["bbox"] if first_page["blocks"] else None
        return self._build_qa_result(
            question=question, answer=answer, field="Document summary", value=answer,
            page_num=1 if bbox else None, sec_page_num=None, confidence=0.8,
            section_title="Document-derived summary", crop_bbox=bbox,
            snippet_filename=f"crop_session_{session.session_id[:8]}_summary.png" if bbox else None,
        )

    def _build_not_found_result(self, question: str) -> QAResult:
        session = self.current_session
        return QAResult(question, "The uploaded report does not contain this information.", None, None,
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
