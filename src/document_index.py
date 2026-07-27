"""
Document Index Engine using PyMuPDF and TF-IDF.
Extracts searchable text blocks, normalizes coordinates, and builds TF-IDF vector indices.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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


class DocumentIndex:
    """Extracts and indexes text blocks from a PDF file for fast vector search."""

    def __init__(self):
        self.blocks: List[TextBlock] = []
        self.block_texts: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.pages_count: int = 0
        self.pdf_path: Optional[Path] = None

    def build_from_pdf(self, pdf_path: str | Path) -> None:
        """Extract all text blocks from the PDF and generate TF-IDF embeddings."""
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        self.blocks.clear()
        self.block_texts.clear()

        logger.info(f"Indexing PDF document: {self.pdf_path}")
        with fitz.open(self.pdf_path) as doc:
            self.pages_count = len(doc)
            for page_index in range(self.pages_count):
                page = doc[page_index]
                page_num = page_index + 1
                page_blocks = self._extract_page_blocks(page, page_num)
                self.blocks.extend(page_blocks)

        self.block_texts = [b.normalized_text for b in self.blocks]

        if self.block_texts:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b"
            )
            self.embeddings = self.vectorizer.fit_transform(self.block_texts).toarray()
            logger.info(f"Successfully indexed {len(self.blocks)} text blocks across {self.pages_count} pages.")
        else:
            self.vectorizer = None
            self.embeddings = None
            logger.warning(f"No text blocks found in PDF: {self.pdf_path}")

    def get_top_k_blocks(self, query: str, k: int = 5) -> List[TextBlock]:
        """
        Retrieve Top K most relevant text blocks using TF-IDF cosine similarity.

        Args:
            query (str): Question or search keywords.
            k (int): Number of top blocks to return (default 5).

        Returns:
            List[TextBlock]: Up to k most relevant text blocks.
        """
        if not self.blocks:
            return []

        if len(self.blocks) <= k or not self.vectorizer or self.embeddings is None:
            return self.blocks[:k]

        from sklearn.metrics.pairwise import cosine_similarity
        normalized_q = " ".join(query.casefold().split())
        query_tokens = [t for t in re.findall(r"[\w']+", normalized_q) if t and t not in STOP_WORDS]
        search_str = " ".join(query_tokens) if query_tokens else normalized_q

        try:
            query_vec = self.vectorizer.transform([search_str])
            scores = cosine_similarity(query_vec, self.embeddings)[0]
            ranked_indices = np.argsort(scores)[::-1]
            top_indices = ranked_indices[:k]
            return [self.blocks[i] for i in top_indices]
        except Exception as e:
            logger.warning(f"Error computing TF-IDF similarity for top-k blocks: {e}")
            return self.blocks[:k]

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
