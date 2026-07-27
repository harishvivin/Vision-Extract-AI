"""
Document-Grounded QA Engine.
Indexes uploaded PDF documents on demand, matches natural language queries against indexed text blocks,
and generates exact visual screenshot crops. Answers are strictly grounded in the uploaded document.
"""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from config import OUTPUTS_DIR
from src.document_index import DocumentIndex, TextBlock
from src.question_parser import QuestionParser, ParsedQuestion
from src.block_matcher import BlockMatcher, MatchResult

logger = logging.getLogger("qa_engine")

NOT_FOUND_ANSWER = "The uploaded report does not contain this information."


@dataclass
class QAResult:
    """Structured QA query response container."""
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
    """Isolated session state for a single uploaded PDF document."""
    session_id: str
    document_name: str
    pdf_path: Path
    index: DocumentIndex


class DocumentQAEngine:
    """Answers user questions strictly from the currently uploaded PDF document."""

    def __init__(self, outputs_dir: Path = OUTPUTS_DIR):
        self.outputs_dir = Path(outputs_dir)
        self.snippets_dir = self.outputs_dir / "qa_snippets"
        self.snippets_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[DocumentSession] = None
        self.parser = QuestionParser()

    def purge_and_create_session(self, pdf_path: str | Path, document_name: Optional[str] = None) -> str:
        """
        Purge existing session data and snippets, then build a fresh index for the uploaded PDF.

        Args:
            pdf_path (str | Path): Path to the uploaded PDF file.
            document_name (Optional[str]): Friendly display name for document.

        Returns:
            str: Newly generated session ID.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Uploaded PDF document not found at: {pdf_path}")

        # Purge previous session evidence snippets
        self._purge_snippets()
        self.current_session = None

        session_id = str(uuid.uuid4())
        doc_name = document_name or pdf_path.name
        logger.info(f"Creating fresh QA session [{session_id[:8]}] for document '{doc_name}' ({pdf_path})")

        index = DocumentIndex()
        index.build_from_pdf(pdf_path)

        self.current_session = DocumentSession(
            session_id=session_id,
            document_name=doc_name,
            pdf_path=pdf_path,
            index=index
        )

        return session_id

    def ask(self, question: str, session_id: Optional[str] = None) -> QAResult:
        """
        Answer a question using only the current active document session.

        Args:
            question (str): User question text.
            session_id (Optional[str]): Active session ID validation check.

        Returns:
            QAResult: Struct containing answer, page, bounding box, and snippet details.
        """
        if not self.current_session:
            logger.warning(f"QA engine invoked without active document session for query: '{question}'")
            return self._build_not_found_result(question)

        session = self.current_session
        if session_id and session_id != session.session_id:
            raise ValueError("Provided session_id does not match current active document session.")

        parsed_q = self.parser.parse(question)

        # 1. Handle Summary Request
        if parsed_q.is_summary_request:
            return self._generate_summary_result(question)

        # 2. Handle Field / Information Lookup
        matcher = BlockMatcher(session.index)
        match_result = matcher.match(question, parsed_q.keywords)

        if not match_result or not match_result.answer_value:
            logger.info(f"Answer for query '{question}' not found in active document.")
            return self._build_not_found_result(question)

        block = match_result.block
        snippet_filename = f"crop_session_{session.session_id[:8]}_p{block.page_number}_{uuid.uuid4().hex[:8]}.png"

        return self._build_qa_result(
            question=question,
            answer=match_result.answer_value,
            field="Document Field",
            value=match_result.answer_value,
            page_num=block.page_number,
            sec_page_num=None,
            confidence=match_result.confidence,
            section_title=self._derive_section_title(block.text),
            crop_bbox=block.bbox,
            snippet_filename=snippet_filename
        )

    def _generate_summary_result(self, question: str) -> QAResult:
        """Generate a concise summary from the active uploaded document."""
        assert self.current_session is not None
        session = self.current_session
        excerpts: List[str] = []
        seen = set()

        for block in session.index.blocks:
            clean_text = " ".join(block.text.replace("\u00a0", " ").split()).strip()
            key = clean_text.casefold()
            if key not in seen and len(clean_text) >= 30 and not self._is_metadata_line(clean_text):
                excerpts.append(clean_text)
                seen.add(key)
            if len(excerpts) >= 4:
                break

        if excerpts:
            summary_text = " ".join(excerpts[:4])
        else:
            summary_text = f"Document '{session.document_name}' contains {session.index.pages_count} pages."

        first_block = session.index.blocks[0] if session.index.blocks else None
        crop_bbox = first_block.bbox if first_block else [0.1, 0.1, 0.9, 0.3]
        page_num = first_block.page_number if first_block else 1
        snippet_filename = f"crop_session_{session.session_id[:8]}_summary.png"

        return self._build_qa_result(
            question=question,
            answer=summary_text,
            field="Document Summary",
            value=summary_text,
            page_num=page_num,
            sec_page_num=None,
            confidence=0.90,
            section_title="Document Summary",
            crop_bbox=crop_bbox,
            snippet_filename=snippet_filename
        )

    @staticmethod
    def _is_metadata_line(text: str) -> bool:
        lower = text.casefold()
        return "processed" in lower or "indexed" in lower or "metadata" in lower

    def _build_not_found_result(self, question: str) -> QAResult:
        session = self.current_session
        return QAResult(
            question=question,
            answer=NOT_FOUND_ANSWER,
            field=None,
            value=None,
            page_number=None,
            secondary_page_number=None,
            confidence=0.0,
            section_title="No matching evidence",
            bounding_box=None,
            snippet_filename=None,
            snippet_path=None,
            session_id=session.session_id if session else "none",
            document_name=session.document_name if session else "none"
        )

    def _build_qa_result(
        self,
        question: str,
        answer: str,
        field: Optional[str],
        value: Optional[str],
        page_num: Optional[int],
        sec_page_num: Optional[int],
        confidence: float,
        section_title: str,
        crop_bbox: Optional[List[float]],
        snippet_filename: Optional[str]
    ) -> QAResult:
        assert self.current_session is not None
        snippet_path: Optional[Path] = None

        if self._valid_bbox(crop_bbox) and page_num is not None and snippet_filename:
            snippet_path = self.snippets_dir / snippet_filename
            self._crop_snippet_from_pdf(self.current_session.pdf_path, page_num, crop_bbox, snippet_path)
        else:
            crop_bbox = None
            snippet_filename = None

        return QAResult(
            question=question,
            answer=answer,
            field=field,
            value=value,
            page_number=page_num,
            secondary_page_number=sec_page_num,
            confidence=confidence,
            section_title=section_title,
            bounding_box=crop_bbox,
            snippet_filename=snippet_filename,
            snippet_path=str(snippet_path) if snippet_path and snippet_path.exists() else None,
            session_id=self.current_session.session_id,
            document_name=self.current_session.document_name
        )

    def _crop_snippet_from_pdf(self, pdf_path: Path, page_num: int, bbox: List[float], output_path: Path) -> None:
        """Render high-DPI page pixmap and crop exact bounding box region with visual outline."""
        try:
            with fitz.open(pdf_path) as doc:
                if not 1 <= page_num <= len(doc):
                    return
                page = doc[page_num - 1]
                pix = page.get_pixmap(dpi=250)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            width, height = image.size
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            padding = 16
            crop_box = (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(width, x2 + padding),
                min(height, y2 + padding)
            )

            cropped = image.crop(crop_box)
            draw = ImageDraw.Draw(cropped)
            draw.rectangle([(2, 2), (cropped.width - 3, cropped.height - 3)], outline="#10b981", width=5)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path, format="PNG")
            logger.info(f"Saved evidence crop snippet to: {output_path}")

        except Exception as e:
            logger.exception(f"Failed to generate evidence crop snippet: {e}")

    def _purge_snippets(self) -> None:
        if self.snippets_dir.exists():
            for f in self.snippets_dir.glob("*.png"):
                try:
                    f.unlink()
                except OSError:
                    pass

    @staticmethod
    def _derive_section_title(text: str) -> str:
        first_line = text.strip().splitlines()[0] if text.strip() else "Document Evidence"
        return first_line[:50] + "..." if len(first_line) > 50 else first_line

    @staticmethod
    def _valid_bbox(bbox: Optional[List[float]]) -> bool:
        return bool(bbox and len(bbox) == 4 and 0 <= bbox[0] < bbox[2] <= 1.0 and 0 <= bbox[1] < bbox[3] <= 1.0)

    def get_sample_questions(self) -> List[Dict[str, Any]]:
        """Return generic document sample questions."""
        return [
            {"icon": "👤", "question": "What is the patient's name?", "tag": "Patient", "page": 1},
            {"icon": "🏥", "question": "What is the hospital or lab name?", "tag": "Hospital", "page": 1},
            {"icon": "📋", "question": "What is the diagnosis?", "tag": "Diagnosis", "page": 1},
            {"icon": "🩸", "question": "What is the hemoglobin level?", "tag": "Lab Result", "page": 1},
            {"icon": "🧪", "question": "What is the creatinine level?", "tag": "Lab Result", "page": 1},
            {"icon": "📊", "question": "What is the HbA1c percentage?", "tag": "Lab Result", "page": 1},
            {"icon": "💓", "question": "What is the blood pressure reading?", "tag": "Vitals", "page": 1},
            {"icon": "📝", "question": "Summarize this report.", "tag": "Summary", "page": 1},
        ]
