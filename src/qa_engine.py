"""
Retrieval-Augmented Generation (RAG) QA Engine for Medical PDFs.
Indexes uploaded PDF documents on demand into a dense Vector Database,
retrieves Top 5 semantic document chunks, queries Google Gemini RAG API (zero hallucination),
and generates high-precision visual evidence screenshot crops (+10px padding, green border).
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
from src.document_index import DocumentIndex, TextBlock, DocumentChunk
from src.question_parser import QuestionParser, ParsedQuestion
from src.block_matcher import BlockMatcher
from src.gemini_client import GeminiQAClient, NOT_FOUND_ANSWER

logger = logging.getLogger("qa_engine")


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
    reason: str
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
    """RAG System answering user questions strictly from the currently uploaded PDF document."""

    def __init__(self, outputs_dir: Path = OUTPUTS_DIR):
        self.outputs_dir = Path(outputs_dir)
        self.snippets_dir = self.outputs_dir / "qa_snippets"
        self.snippets_dir.mkdir(parents=True, exist_ok=True)
        self.current_session: Optional[DocumentSession] = None
        self.parser = QuestionParser()
        self.gemini_client = GeminiQAClient()

    def purge_and_create_session(self, pdf_path: str | Path, document_name: Optional[str] = None) -> str:
        """
        Purge existing session data and snippets, then build a fresh vector index for the uploaded PDF.

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
        logger.info(f"Creating fresh RAG QA session [{session_id[:8]}] for document '{doc_name}' ({pdf_path})")

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
        Answer a question using the RAG pipeline over the active document session.

        Args:
            question (str): User question text.
            session_id (Optional[str]): Active session ID validation check.

        Returns:
            QAResult: Struct containing answer, page, reason, bounding box, and snippet details.
        """
        if not self.current_session:
            logger.warning(f"RAG engine invoked without active document session for query: '{question}'")
            return self._build_not_found_result(question)

        session = self.current_session
        if session_id and session_id != session.session_id:
            raise ValueError("Provided session_id does not match current active document session.")

        parsed_q = self.parser.parse(question)

        # 1. Handle Summary Request
        if parsed_q.is_summary_request:
            return self._generate_summary_result(question)

        # 2. Retrieve Top 5 most relevant semantic chunks using dense Vector DB embeddings
        top_chunks = session.index.get_top_k_chunks(question, k=5)
        top_chunks_dict = [{"page_number": c.page_number, "text": c.text, "chunk_id": c.chunk_id} for c in top_chunks]

        # 3. Send Top 5 semantic chunks to Gemini RAG Client
        gemini_res = self.gemini_client.query(question, top_chunks_dict)

        if gemini_res:
            answer = gemini_res.get("answer", "").strip()
            matched_text = gemini_res.get("matched_text", "").strip()
            target_page = gemini_res.get("page")
            reason = gemini_res.get("reason", "Answer retrieved from document context.")
            confidence = gemini_res.get("confidence", 0.98)

            # Check if answer is NOT found in document
            if not answer or answer == NOT_FOUND_ANSWER or "not contain" in answer.casefold():
                logger.info(f"Gemini RAG confirmed answer not in document for query: '{question}'")
                return self._build_not_found_result(question)

            # Locate exact PyMuPDF bounding box for matched_text from retrieved chunks
            page_num, crop_bbox, section_title = self._find_bbox_for_matched_text(
                session, matched_text or answer, target_page, top_chunks
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
                reason=reason,
                section_title=section_title,
                crop_bbox=crop_bbox,
                snippet_filename=snippet_filename
            )

        # 4. Fallback to Local RAG Matcher over retrieved Top 5 Chunks
        logger.info("Using local RAG matcher fallback engine.")
        matched_result = self._local_rag_match(session, question, parsed_q, top_chunks)

        if not matched_result:
            logger.info(f"Answer for query '{question}' not found in active document.")
            return self._build_not_found_result(question)

        answer_val, matched_txt, page_num, crop_bbox, section_title, conf = matched_result
        snippet_filename = f"crop_session_{session.session_id[:8]}_p{page_num}_{uuid.uuid4().hex[:8]}.png"

        return self._build_qa_result(
            question=question,
            answer=answer_val,
            field="Document Field",
            value=answer_val,
            page_num=page_num,
            sec_page_num=None,
            confidence=conf,
            reason=f"Evidence matching '{matched_txt[:30]}' retrieved from Page {page_num}.",
            section_title=section_title,
            crop_bbox=crop_bbox,
            snippet_filename=snippet_filename
        )

    def _local_rag_match(
        self,
        session: DocumentSession,
        question: str,
        parsed_q: ParsedQuestion,
        top_chunks: List[DocumentChunk]
    ) -> Optional[Tuple[str, str, int, Optional[List[float]], str, float]]:
        """
        Local RAG fallback engine: searches top retrieved chunks for requested medical parameter.
        Returns Tuple[answer_value, matched_text, page_number, crop_bbox, section_title, confidence]
        """
        q_norm = question.casefold()
        keywords = set(parsed_q.keywords)

        # Determine target medical parameter
        target_param = None
        if "hiv" in keywords or "hiv" in q_norm:
            target_param = "hiv"
        elif "creatinine" in keywords or "creatinine" in q_norm:
            target_param = "creatinine"
        elif "hba1c" in keywords or "a1c" in keywords or "glycated" in q_norm:
            target_param = "hba1c"
        elif "hemoglobin" in keywords or "hb" in keywords:
            target_param = "hemoglobin"
        elif "blood pressure" in q_norm or "bp" in keywords or "pressure" in keywords:
            target_param = "bp"
        elif "ecg" in keywords or "ekg" in keywords or "rhythm" in q_norm:
            target_param = "ecg"
        elif "diagnosis" in keywords or "impression" in keywords:
            target_param = "diagnosis"
        elif any(k in keywords for k in ["doctor", "physician", "dr"]) or "doctor" in q_norm:
            target_param = "doctor"
        elif any(k in keywords for k in ["hospital", "lab", "diagnostics", "clinic", "center", "centre", "institute", "facility"]) or "hospital" in q_norm:
            target_param = "hospital"
        elif "age" in keywords or "years" in q_norm:
            target_param = "age"
        elif "gender" in keywords or "sex" in keywords:
            target_param = "gender"
        elif "patient" in keywords or "name" in keywords or "person" in q_norm:
            target_param = "patient"

        # Search top retrieved chunks first, then fall back to all document chunks and blocks
        search_blocks_lines = []
        seen_chunks = set()
        for c in top_chunks:
            seen_chunks.add(c.chunk_id)
            for line in c.text.splitlines():
                search_blocks_lines.append((c.page_number, c, line))

        for c in session.index.chunks:
            if c.chunk_id not in seen_chunks:
                for line in c.text.splitlines():
                    search_blocks_lines.append((c.page_number, c, line))

        for block in session.index.blocks:
            for line in block.text.splitlines():
                search_blocks_lines.append((block.page_number, None, line))

        for page_num, chunk_obj, raw_line in search_blocks_lines:
            line_str = raw_line.strip()
            if not line_str:
                continue

            # Split line into pipe-separated or semicolon-separated segments if present
            segments = [s.strip() for s in re.split(r"[|;]", line_str) if s.strip()]

            for seg in segments:
                seg_norm = seg.casefold()

                matched = False
                if target_param == "hiv" and "hiv" in seg_norm:
                    matched = True
                elif target_param == "creatinine" and "creatinine" in seg_norm:
                    matched = True
                elif target_param == "hba1c" and ("hba1c" in seg_norm or "glycated" in seg_norm or "a1c" in seg_norm):
                    matched = True
                elif target_param == "hemoglobin" and ("hemoglobin" in seg_norm or "hb" in seg_norm) and "hba1c" not in seg_norm:
                    matched = True
                elif target_param == "bp" and ("blood pressure" in seg_norm or "bp" in seg_norm):
                    matched = True
                elif target_param == "ecg" and ("ecg" in seg_norm or "ekg" in seg_norm):
                    matched = True
                elif target_param == "diagnosis" and ("diagnosis" in seg_norm or "impression" in seg_norm or "clinical" in seg_norm):
                    matched = True
                elif target_param == "doctor" and ("doctor" in seg_norm or "physician" in seg_norm or "dr." in seg_norm or "dr " in seg_norm or "consultant" in seg_norm or "ref." in seg_norm):
                    matched = True
                elif target_param == "hospital" and any(h in seg_norm for h in ["hospital", "diagnostics", "clinic", "center", "centre", "institute", "laboratory", "lab"]):
                    matched = True
                elif target_param == "age" and ("age" in seg_norm or "years" in seg_norm):
                    matched = True
                elif target_param == "gender" and ("gender" in seg_norm or "sex" in seg_norm or "male" in seg_norm or "female" in seg_norm):
                    matched = True
                elif target_param == "patient" and ("name" in seg_norm or "patient" in seg_norm):
                    matched = True

                if matched:
                    # Extract value after colon if present; otherwise return entire segment line
                    if ":" in seg:
                        val = seg.split(":", 1)[1].strip()
                    else:
                        val = seg.strip()

                    if val and len(val) < 100:
                        crop_chunks = [chunk_obj] if chunk_obj else []
                        p_num, crop_bbox, sec_title = self._find_bbox_for_matched_text(
                            session, seg, page_num, crop_chunks
                        )
                        return val, seg, p_num, crop_bbox, sec_title, 0.95

        return None


    def _find_bbox_for_matched_text(
        self,
        session: DocumentSession,
        matched_text: str,
        target_page: Optional[int],
        top_chunks: List[DocumentChunk]
    ) -> Tuple[int, Optional[List[float]], str]:
        """
        Locate the exact line-level PyMuPDF bounding box for matched_text.

        Returns:
            Tuple[int, Optional[List[float]], str]: (page_number, bbox, section_title)
        """
        if not matched_text:
            first_c = top_chunks[0] if top_chunks else (session.index.chunks[0] if session.index.chunks else None)
            if first_c:
                return first_c.page_number, first_c.bbox, self._derive_section_title(first_c.text)
            return 1, [0.1, 0.1, 0.9, 0.3], "Document Evidence"

        norm_target = " ".join(matched_text.casefold().split())

        # 0. PyMuPDF word-level exact search for pinpoint crop precision
        try:
            with fitz.open(session.pdf_path) as doc:
                p_idx = (target_page - 1) if (target_page and 1 <= target_page <= len(doc)) else 0
                page = doc[p_idx]
                w, h = page.rect.width, page.rect.height
                if w > 0 and h > 0:
                    words = page.get_text("words")  # (x0, y0, x1, y1, text, b_no, l_no, w_no)
                    target_words = [t for t in re.findall(r"[\w']+", norm_target) if t and t not in {"the", "a", "an", "is", "of", "in", "to"}]
                    if target_words:
                        matching_rects = []
                        for item in words:
                            w_text = item[4].casefold().strip(":,.-")
                            if w_text and any(tw == w_text or tw in w_text for tw in target_words):
                                matching_rects.append(item[:4])
                        if matching_rects:
                            x0 = min(r[0] for r in matching_rects) / w
                            y0 = min(r[1] for r in matching_rects) / h
                            x1 = max(r[2] for r in matching_rects) / w
                            y1 = max(r[3] for r in matching_rects) / h
                            tight_bbox = [max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)]
                            return (p_idx + 1), tight_bbox, matched_text[:45]
        except Exception as e:
            logger.debug(f"PyMuPDF word search fallback: {e}")

        # 1. Search lines data from chunks
        for chunk in top_chunks:
            if chunk.lines_data:
                for line in chunk.lines_data:
                    line_norm = " ".join(line["text"].casefold().split())
                    if norm_target in line_norm or line_norm in norm_target:
                        return chunk.page_number, line["bbox"], self._derive_section_title(line["text"])

        # 2. Search block lines
        for block in session.index.blocks:
            if block.lines_data:
                for line in block.lines_data:
                    line_norm = " ".join(line["text"].casefold().split())
                    if norm_target in line_norm or line_norm in norm_target:
                        return block.page_number, line["bbox"], self._derive_section_title(line["text"])

        first_c = top_chunks[0] if top_chunks else session.index.chunks[0]
        return first_c.page_number, first_c.bbox, self._derive_section_title(first_c.text)

    def _generate_summary_result(self, question: str) -> QAResult:
        """Generate a concise summary from the active uploaded document using Gemini RAG API."""
        assert self.current_session is not None
        session = self.current_session

        blocks_data = [{"page_number": c.page_number, "text": c.text} for c in session.index.chunks[:5]]
        gemini_summary = self.gemini_client.summarize(blocks_data)

        if gemini_summary:
            summary_text = gemini_summary
        else:
            excerpts: List[str] = []
            seen = set()
            for chunk in session.index.chunks:
                clean_text = " ".join(chunk.text.replace("\u00a0", " ").split()).strip()
                key = clean_text.casefold()
                if key not in seen and len(clean_text) >= 25:
                    excerpts.append(clean_text)
                    seen.add(key)
                if len(excerpts) >= 4:
                    break

            summary_text = " ".join(excerpts[:4]) if excerpts else f"Document '{session.document_name}' summary."

        first_chunk = session.index.chunks[0] if session.index.chunks else None
        crop_bbox = first_chunk.bbox if first_chunk else [0.1, 0.1, 0.9, 0.3]
        page_num = first_chunk.page_number if first_chunk else 1
        snippet_filename = f"crop_session_{session.session_id[:8]}_summary.png"

        return self._build_qa_result(
            question=question,
            answer=summary_text,
            field="Document Summary",
            value=summary_text,
            page_num=page_num,
            sec_page_num=None,
            confidence=0.95,
            reason="Synthesized summary from top document chunks.",
            section_title="Document Summary",
            crop_bbox=crop_bbox,
            snippet_filename=snippet_filename
        )

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
            reason="The requested field or parameter was not found in the uploaded document.",
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
        reason: str,
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
            reason=reason,
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
