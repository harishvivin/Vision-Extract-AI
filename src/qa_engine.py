"""
Document-Grounded QA Engine.
Indexes uploaded PDF documents on demand, retrieves Top 5 text blocks via TF-IDF search,
queries Google Gemini API for document understanding, and generates 10px expanded visual screenshot crops.
Answers are strictly grounded in the uploaded document.
"""

import logging
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import io
import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from config import OUTPUTS_DIR
from src.document_index import DocumentIndex, TextBlock
from src.question_parser import QuestionParser, ParsedQuestion
from src.block_matcher import BlockMatcher, MatchResult
from src.gemini_client import GeminiQAClient

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
        self.gemini_client = GeminiQAClient()

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

        # 2. Retrieve Top 5 most relevant text blocks using TF-IDF search
        top_blocks_obj = session.index.get_top_k_blocks(question, k=5)
        top_blocks = [{"page_number": b.page_number, "text": b.text} for b in top_blocks_obj]

        # 3. Send Top 5 text blocks and question to Gemini
        gemini_res = self.gemini_client.query(question, top_blocks)

        if gemini_res:
            answer = gemini_res.get("answer", "").strip()
            matched_text = gemini_res.get("matched_text", "").strip()
            target_page = gemini_res.get("page")
            confidence = gemini_res.get("confidence", 0.98)

            # Check if answer is NOT found in document
            if not answer or answer == NOT_FOUND_ANSWER or "not contain" in answer.casefold():
                logger.info(f"Gemini confirmed answer not in document for query: '{question}'")
                return self._build_not_found_result(question)

            # Locate exact PyMuPDF bounding box for matched_text
            page_num, crop_bbox, section_title = self._find_bbox_for_matched_text(
                session, matched_text or answer, target_page, top_blocks_obj
            )

            snippet_filename = f"crop_session_{session.session_id[:8]}_p{page_num}_{uuid.uuid4().hex[:8]}.png"

            return self._build_qa_result(
                question=question,
                answer=answer,
                field="Document Field",
                value=answer,
                page_num=page_num,
                sec_page_num=None,
                confidence=confidence,
                section_title=section_title,
                crop_bbox=crop_bbox,
                snippet_filename=snippet_filename
            )

        # 4. Fallback to local BlockMatcher if Gemini API key is missing or call fails
        logger.info("Using local BlockMatcher fallback engine.")
        matcher = BlockMatcher(session.index)
        match_result = matcher.match(question, parsed_q.keywords)

        if not match_result or not match_result.answer_value:
            logger.info(f"Answer for query '{question}' not found in active document.")
            return self._build_not_found_result(question)

        block = match_result.block
        page_num, crop_bbox, section_title = self._find_bbox_for_matched_text(
            session, match_result.answer_value, block.page_number, [block]
        )

        snippet_filename = f"crop_session_{session.session_id[:8]}_p{page_num}_{uuid.uuid4().hex[:8]}.png"

        return self._build_qa_result(
            question=question,
            answer=match_result.answer_value,
            field="Document Field",
            value=match_result.answer_value,
            page_num=page_num,
            sec_page_num=None,
            confidence=match_result.confidence,
            section_title=section_title,
            crop_bbox=crop_bbox,
            snippet_filename=snippet_filename
        )

    def _find_bbox_for_matched_text(
        self,
        session: DocumentSession,
        matched_text: str,
        target_page: Optional[int],
        top_blocks: List[TextBlock]
    ) -> Tuple[int, Optional[List[float]], str]:
        """
        Locate the exact line-level or block-level PyMuPDF bounding box for matched_text.

        Returns:
            Tuple[int, Optional[List[float]], str]: (page_number, bbox, section_title)
        """
        if not matched_text:
            first_b = top_blocks[0] if top_blocks else session.index.blocks[0]
            return first_b.page_number, first_b.bbox, self._derive_section_title(first_b.text)

        norm_target = " ".join(matched_text.casefold().split())

        # Prioritize blocks on target_page if specified
        search_blocks = list(session.index.blocks)
        if target_page is not None:
            search_blocks.sort(key=lambda b: 0 if b.page_number == target_page else 1)

        # 1. Search line-level exact or substring matches
        for block in search_blocks:
            if block.lines_data:
                matching_line_boxes = []
                for line in block.lines_data:
                    line_norm = " ".join(line["text"].casefold().split())
                    if norm_target in line_norm or line_norm in norm_target:
                        matching_line_boxes.append(line["bbox"])
                
                if matching_line_boxes:
                    merged_line_box = matching_line_boxes[0]
                    for lb in matching_line_boxes[1:]:
                        merged_line_box = [
                            min(merged_line_box[0], lb[0]),
                            min(merged_line_box[1], lb[1]),
                            max(merged_line_box[2], lb[2]),
                            max(merged_line_box[3], lb[3])
                        ]
                    return block.page_number, merged_line_box, self._derive_section_title(block.text)

        # 2. Search token overlap in lines
        target_tokens = set(re.findall(r"[\w']+", norm_target)) - {"the", "a", "an", "is", "of", "in", "to", "value", "level"}
        if target_tokens:
            best_line_box = None
            best_line_overlap = 0
            best_block = None
            for block in search_blocks:
                if block.lines_data:
                    for line in block.lines_data:
                        line_tokens = set(re.findall(r"[\w']+", line["text"].casefold()))
                        overlap = len(target_tokens.intersection(line_tokens))
                        if overlap > best_line_overlap:
                            best_line_overlap = overlap
                            best_line_box = line["bbox"]
                            best_block = block
            if best_line_box and best_block and best_line_overlap >= 1:
                return best_block.page_number, best_line_box, self._derive_section_title(best_block.text)

        # 3. Fallback to block bbox
        for block in search_blocks:
            if norm_target in block.normalized_text or block.normalized_text in norm_target:
                return block.page_number, block.bbox, self._derive_section_title(block.text)

        first_block = top_blocks[0] if top_blocks else session.index.blocks[0]
        return first_block.page_number, first_block.bbox, self._derive_section_title(first_block.text)

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
        """Render high-DPI page pixmap and crop exact bounding box region expanded by 10px with green outline."""
        try:
            with fitz.open(pdf_path) as doc:
                if not 1 <= page_num <= len(doc):
                    return
                page = doc[page_num - 1]
                pix = page.get_pixmap(dpi=150)
                image = Image.open(io.BytesIO(pix.tobytes("png")))

            width, height = image.size
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            # Expand bounding box by exactly 10 pixels
            padding = 10
            crop_box = (
                max(0, x1 - padding),
                max(0, y1 - padding),
                min(width, x2 + padding),
                min(height, y2 + padding)
            )

            cropped = image.crop(crop_box)
            draw = ImageDraw.Draw(cropped)
            draw.rectangle([(1, 1), (cropped.width - 2, cropped.height - 2)], outline="#10b981", width=4)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cropped.save(output_path, format="PNG", compress_level=1)
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

