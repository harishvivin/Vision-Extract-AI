"""
Document Index & Intelligent Semantic Chunker using PyMuPDF and Dense Vector Store.
Extracts page layout, text blocks, and line coordinates, chunks documents semantically,
and generates dense vector embeddings stored in a VectorDatabase.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import numpy as np

from src.vector_store import VectorDatabase, DocumentChunk

logger = logging.getLogger("document_index")

STOP_WORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does", "for",
    "from", "has", "have", "i", "in", "is", "it", "me", "of", "on", "or", "please", "show",
    "tell", "the", "this", "to", "value", "what", "where", "which", "who", "with"
})


@dataclass
class TextBlock:
    """Represents a text block extracted from a PDF page."""
    block_id: str
    page_number: int
    bbox: List[float]  # Normalized [x1, y1, x2, y2] (0.0 to 1.0)
    raw_bbox: List[float]  # Point coordinates [x1, y1, x2, y2]
    text: str
    normalized_text: str
    tokens: List[str]
    lines_data: Optional[List[Dict[str, Any]]] = None  # Per-line details: text, raw_bbox, bbox


class SemanticChunker:
    """Intelligently chunks PyMuPDF text blocks into cohesive semantic sections without fixed-character cutting."""

    @staticmethod
    def chunk_blocks(blocks: List[TextBlock], page_number: int) -> List[DocumentChunk]:
        """
        Group page blocks into semantic chunks preserving headings, key-value pairs, and reference ranges.
        """
        if not blocks:
            return []

        chunks: List[DocumentChunk] = []
        current_lines: List[Dict[str, Any]] = []
        current_texts: List[str] = []
        current_raw_bbox: Optional[List[float]] = None
        current_heading: str = ""

        def merge_bbox(b1: Optional[List[float]], b2: List[float]) -> List[float]:
            if not b1:
                return list(b2)
            return [
                min(b1[0], b2[0]),
                min(b1[1], b2[1]),
                max(b1[2], b2[2]),
                max(b1[3], b2[3])
            ]

        def finalize_chunk():
            nonlocal current_lines, current_texts, current_raw_bbox, current_heading
            if not current_texts or not current_raw_bbox:
                return

            full_text = "\n".join(current_texts)
            norm_text = " ".join(full_text.casefold().split())
            chunk_id = f"p{page_number}_c{len(chunks)+1}"

            # Calculate normalized bounding box (will be computed during page rendering)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                page_number=page_number,
                text=full_text,
                normalized_text=norm_text,
                bbox=[0.0, 0.0, 1.0, 1.0],  # Updated per page dimensions
                raw_bbox=list(current_raw_bbox),
                lines_data=list(current_lines),
                section_heading=current_heading
            ))

            current_lines = []
            current_texts = []
            current_raw_bbox = None

        for b in blocks:
            # Detect section heading
            is_heading = False
            first_line = b.text.strip().splitlines()[0] if b.text.strip() else ""
            if first_line and (first_line.isupper() or len(first_line) < 40 or ":" not in first_line):
                if any(h in first_line.casefold() for h in [
                    "report", "hospital", "patient", "clinical", "biochemistry", "haematology",
                    "serology", "diagnostic", "impression", "investigation", "department", "doctor",
                    "pathology", "radiology", "ecg", "vitals", "summary"
                ]):
                    is_heading = True
                    current_heading = first_line

            # If new heading and accumulated enough content, split semantic chunk
            if is_heading and len(current_texts) >= 3:
                finalize_chunk()

            current_texts.append(b.text)
            current_raw_bbox = merge_bbox(current_raw_bbox, b.raw_bbox)
            if b.lines_data:
                current_lines.extend(b.lines_data)

            # If single block is very substantial (e.g. multi-line lab test table), check size
            if len("\n".join(current_texts)) > 800:
                finalize_chunk()

        if current_texts:
            finalize_chunk()

        return chunks


class DocumentIndex:
    """Extracts and indexes semantic chunks from a PDF file using a dense VectorDatabase."""

    def __init__(self):
        self.blocks: List[TextBlock] = []
        self.chunks: List[DocumentChunk] = []
        self.vector_db = VectorDatabase()
        self.pages_count: int = 0
        self.pdf_path: Optional[Path] = None

    def build_from_pdf(self, pdf_path: str | Path) -> None:
        """Extract text blocks, semantically chunk document, and store in vector database."""
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self.blocks.clear()
        self.chunks.clear()
        self.vector_db.clear()

        logger.info(f"Indexing PDF document with PyMuPDF & Vector DB: {self.pdf_path}")
        with fitz.open(self.pdf_path) as doc:
            self.pages_count = len(doc)
            for page_index in range(self.pages_count):
                page = doc[page_index]
                page_num = page_index + 1
                page_blocks = self._extract_page_blocks(page, page_num)
                self.blocks.extend(page_blocks)

                # Intelligently chunk page blocks
                page_chunks = SemanticChunker.chunk_blocks(page_blocks, page_num)

                # Normalize raw bboxes per page dimensions
                w, h = page.rect.width, page.rect.height
                if w > 0 and h > 0:
                    for c in page_chunks:
                        c.bbox = [
                            max(0.0, min(1.0, c.raw_bbox[0] / w)),
                            max(0.0, min(1.0, c.raw_bbox[1] / h)),
                            max(0.0, min(1.0, c.raw_bbox[2] / w)),
                            max(0.0, min(1.0, c.raw_bbox[3] / h))
                        ]

                self.chunks.extend(page_chunks)

        if self.chunks:
            self.vector_db.add_chunks(self.chunks)
            logger.info(f"Successfully indexed {len(self.chunks)} semantic chunks ({len(self.blocks)} blocks) across {self.pages_count} pages.")
        else:
            logger.warning(f"No text blocks found in PDF: {self.pdf_path}")

    def get_top_k_chunks(self, query: str, k: int = 5) -> List[DocumentChunk]:
        """
        Retrieve Top K most relevant semantic chunks from the vector database.

        Args:
            query (str): Question text.
            k (int): Number of top chunks to return (default 5).

        Returns:
            List[DocumentChunk]: Top K relevant document chunks.
        """
        return self.vector_db.query(query, top_k=k)

    def get_top_k_blocks(self, query: str, k: int = 5) -> List[TextBlock]:
        """
        Compatibility method: Returns TextBlock representation of Top K retrieved chunks.
        """
        top_chunks = self.get_top_k_chunks(query, k=k)
        result_blocks: List[TextBlock] = []
        for c in top_chunks:
            result_blocks.append(TextBlock(
                block_id=c.chunk_id,
                page_number=c.page_number,
                bbox=c.bbox,
                raw_bbox=c.raw_bbox,
                text=c.text,
                normalized_text=c.normalized_text,
                tokens=re.findall(r"[\w']+", c.normalized_text),
                lines_data=c.lines_data
            ))
        return result_blocks

    def _extract_page_blocks(self, page: fitz.Page, page_number: int) -> List[TextBlock]:
        """Extract text blocks from a single PyMuPDF page."""
        extracted: List[TextBlock] = []
        page_dict = page.get_text("dict")
        width, height = page.rect.width, page.rect.height

        if width <= 0 or height <= 0:
            return extracted

        block_counter = 0
        for b in page_dict.get("blocks", []):
            if b.get("type") != 0:  # Only text blocks
                continue

            text_lines: List[str] = []
            lines_data: List[Dict[str, Any]] = []
            merged_raw_bbox: Optional[List[float]] = None

            for line in b.get("lines", []):
                line_text = "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                if not line_text:
                    continue
                bbox = line.get("bbox")
                if bbox:
                    raw_line_box = list(bbox)
                    merged_raw_bbox = self._merge_bbox(merged_raw_bbox, raw_line_box)
                    norm_line_box = [
                        max(0.0, min(1.0, raw_line_box[0] / width)),
                        max(0.0, min(1.0, raw_line_box[1] / height)),
                        max(0.0, min(1.0, raw_line_box[2] / width)),
                        max(0.0, min(1.0, raw_line_box[3] / height)),
                    ]
                    lines_data.append({
                        "text": line_text,
                        "raw_bbox": raw_line_box,
                        "bbox": norm_line_box
                    })
                text_lines.append(line_text)

            if not text_lines or not merged_raw_bbox:
                continue

            text = "\n".join(text_lines)
            normalized_text = self._normalize_text(text)
            norm_bbox = [
                max(0.0, min(1.0, merged_raw_bbox[0] / width)),
                max(0.0, min(1.0, merged_raw_bbox[1] / height)),
                max(0.0, min(1.0, merged_raw_bbox[2] / width)),
                max(0.0, min(1.0, merged_raw_bbox[3] / height)),
            ]

            tokens = self._tokenize(normalized_text)
            block_counter += 1
            block_id = f"p{page_number}_b{block_counter}"

            extracted.append(TextBlock(
                block_id=block_id,
                page_number=page_number,
                bbox=norm_bbox,
                raw_bbox=merged_raw_bbox,
                text=text,
                normalized_text=normalized_text,
                tokens=tokens,
                lines_data=lines_data
            ))

        return extracted

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [t for t in re.findall(r"[\w']+", text) if t and t not in STOP_WORDS]

    @staticmethod
    def _merge_bbox(first: Optional[List[float]], second: List[float]) -> List[float]:
        if first is None:
            return list(second)
        return [
            min(first[0], second[0]),
            min(first[1], second[1]),
            max(first[2], second[2]),
            max(first[3], second[3]),
        ]
