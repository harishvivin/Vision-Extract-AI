"""
PDF Reader Engine using PyMuPDF (fitz).
Extracts high-resolution images and question text from each PDF page.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PageData:
    """Container for extracted PDF page data."""
    page_number: int
    raw_text: str
    question_text: str
    page_image: Image.Image
    extracted_photo: Image.Image
    photo_bbox: Tuple[float, float, float, float]  # Normalized coordinates [x1, y1, x2, y2] relative to page image


class PDFReader:
    """PDF Reader class using PyMuPDF to extract text and images from document pages."""

    def __init__(self, dpi: int = 300):
        """
        Initialize PDFReader.

        Args:
            dpi (int): DPI resolution for rendering PDF pages. Defaults to 300.
        """
        self.dpi = dpi

    def extract_all(self, pdf_path: str | Path) -> List[PageData]:
        """
        Extract images and question text from all pages of the given PDF file.

        Args:
            pdf_path (str | Path): Path to the input PDF file.

        Returns:
            List[PageData]: List of PageData objects containing extracted page content.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

        logger.info(f"Opening PDF document: {pdf_path}")
        doc = fitz.open(pdf_path)
        pages_data: List[PageData] = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_data = self._process_page(page, page_idx + 1)
            pages_data.append(page_data)

        doc.close()
        logger.info(f"Successfully processed {len(pages_data)} pages from PDF.")
        return pages_data

    def _process_page(self, page: fitz.Page, page_num: int) -> PageData:
        """
        Process a single PDF page to extract text, full rendering, and target photo.

        Args:
            page (fitz.Page): PyMuPDF page object.
            page_num (int): 1-indexed page number.

        Returns:
            PageData: Extracted page data structure.
        """
        # 1. Extract text
        raw_text = page.get_text("text").strip()
        question_text = self._extract_question_text(raw_text)

        # 2. Render full page image at high DPI
        zoom = self.dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        full_page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 3. Extract the primary photo from page
        photo_img, photo_bbox = self._extract_photo(page, full_page_img)

        logger.debug(f"Page {page_num}: Extracted text length={len(raw_text)}, photo size={photo_img.size}")
        return PageData(
            page_number=page_num,
            raw_text=raw_text,
            question_text=question_text,
            page_image=full_page_img,
            extracted_photo=photo_img,
            photo_bbox=photo_bbox
        )

    def _extract_question_text(self, text: str) -> str:
        """
        Parse raw text to isolate the specific question string.

        Args:
            text (str): Raw text extracted from page.

        Returns:
            str: Cleaned question text.
        """
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        question_lines = []
        is_question = False

        for line in lines:
            if "question" in line.lower() or is_question:
                is_question = True
                question_lines.append(line)
                if line.endswith("."):
                    # Check if next line might be part of filename (e.g. 01_yellow_tulips.png)
                    pass

        if question_lines:
            return " ".join(question_lines)
        return text

    def _extract_photo(self, page: fitz.Page, full_page_img: Image.Image) -> Tuple[Image.Image, Tuple[float, float, float, float]]:
        """
        Extract the photo image directly from page streams or crop from full page image.

        Args:
            page (fitz.Page): PyMuPDF page object.
            full_page_img (Image.Image): Rendered full page image.

        Returns:
            Tuple[Image.Image, Tuple[float, float, float, float]]: Extracted photo image and its bbox [x1, y1, x2, y2].
        """
        images = page.get_images(full=True)
        doc = page.parent

        # Look for largest embedded image on page
        best_img = None
        max_area = 0
        best_xref = None

        for img_info in images:
            xref = img_info[0]
            base_img = doc.extract_image(xref)
            img_bytes = base_img["image"]
            try:
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                area = pil_img.width * pil_img.height
                if area > max_area:
                    max_area = area
                    best_img = pil_img
                    best_xref = xref
            except Exception as e:
                logger.warning(f"Could not load embedded image xref {xref}: {e}")

        # Try finding rect of image on page
        image_rects = page.get_image_rects(best_xref) if best_xref is not None else []
        if image_rects:
            rect = image_rects[0]
            # Convert PyMuPDF rect (72 dpi) to normalized coordinates (0 to 1)
            p_width = page.rect.width
            p_height = page.rect.height
            norm_bbox = (rect.x0 / p_width, rect.y0 / p_height, rect.x1 / p_width, rect.y1 / p_height)
        else:
            norm_bbox = (0.0, 0.1, 1.0, 0.75)  # Reasonable fallback center box

        if best_img is not None and best_img.width > 200 and best_img.height > 200:
            return best_img, norm_bbox

        # Fallback: crop the middle photo region from rendered full page image
        W, H = full_page_img.size
        crop_box = (
            int(norm_bbox[0] * W),
            int(norm_bbox[1] * H),
            int(norm_bbox[2] * W),
            int(norm_bbox[3] * H)
        )
        photo_cropped = full_page_img.crop(crop_box)
        return photo_cropped, norm_bbox
