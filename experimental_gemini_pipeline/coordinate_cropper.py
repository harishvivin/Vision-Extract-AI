"""
Coordinate Cropper Module for Experimental Gemini PDF Pipeline.
Uses pure PyMuPDF (fitz) vector rendering for lossless sub-region cropping.
Zero dependencies on OpenCV, OCR, or OS screenshots.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Union, List, Optional
import fitz  # PyMuPDF

from .config import DEFAULT_DPI

logger = logging.getLogger("coordinate_cropper")


def crop_pdf_region(
    pdf_path: Union[str, Path],
    page_number: int,
    bounding_box: Union[Dict[str, float], List[float]],
    output_path: Union[str, Path],
    padding_pts: float = 10.0,
    dpi: int = DEFAULT_DPI
) -> Path:
    """
    Crop precise region from a PDF page using PyMuPDF vector clip rendering.

    Args:
        pdf_path (Union[str, Path]): Path to target PDF file.
        page_number (int): 1-based page number.
        bounding_box (Union[Dict[str, float], List[float]]): Bounding box coordinates.
            Supports dict (x1, y1, x2, y2 or xmin, ymin, xmax, ymax) or list.
        output_path (Union[str, Path]): Target output file path (.png).
        padding_pts (float): Additional padding in PDF points around target box.
        dpi (int): Rendering resolution DPI (default 300 DPI for ultra-sharp evidence).

    Returns:
        Path: Path to saved PNG crop file.
    """
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if page_number < 1 or page_number > total_pages:
        logger.warning(f"Page number {page_number} out of bounds (1..{total_pages}). Defaulting to page 1.")
        page_number = 1

    page = doc[page_number - 1]
    page_rect = page.rect
    pw, ph = page_rect.width, page_rect.height

    # Parse raw bounding box coordinates
    raw_x1, raw_y1, raw_x2, raw_y2 = _parse_bbox(bounding_box)

    # Determine coordinate scale (Normalized 0..1, Normalized 0..1000, or Absolute Points)
    max_val = max(abs(raw_x1), abs(raw_y1), abs(raw_x2), abs(raw_y2))

    if max_val <= 1.0:
        # Scale 0.0 .. 1.0
        x1 = raw_x1 * pw
        y1 = raw_y1 * ph
        x2 = raw_x2 * pw
        y2 = raw_y2 * ph
    elif max_val <= 1000.0:
        # Scale 0 .. 1000
        x1 = (raw_x1 / 1000.0) * pw
        y1 = (raw_y1 / 1000.0) * ph
        x2 = (raw_x2 / 1000.0) * pw
        y2 = (raw_y2 / 1000.0) * ph
    else:
        # Absolute points
        x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2

    # Ensure x1 < x2 and y1 < y2
    crop_x1 = min(x1, x2) - padding_pts
    crop_y1 = min(y1, y2) - padding_pts
    crop_x2 = max(x1, x2) + padding_pts
    crop_y2 = max(y1, y2) + padding_pts

    # Clamp coordinates strictly within page boundary
    crop_x1 = max(0.0, min(crop_x1, pw))
    crop_y1 = max(0.0, min(crop_y1, ph))
    crop_x2 = max(crop_x1 + 10.0, min(crop_x2, pw))
    crop_y2 = max(crop_y1 + 10.0, min(crop_y2, ph))

    clip_rect = fitz.Rect(crop_x1, crop_y1, crop_x2, crop_y2)
    logger.info(f"Cropping PDF Page {page_number} clip rect: {clip_rect} at {dpi} DPI")

    # Render clipped pixmap using PyMuPDF
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=clip_rect)
    pix.save(output_path)

    doc.close()
    logger.info(f"Saved cropped PNG evidence image to: {output_path}")
    return output_path


def _parse_bbox(bbox: Union[Dict[str, float], List[float]]) -> tuple[float, float, float, float]:
    """Helper to convert dictionary or list bounding box to (x1, y1, x2, y2)."""
    if isinstance(bbox, dict):
        if "x1" in bbox and "y1" in bbox:
            return (
                float(bbox.get("x1", 0)),
                float(bbox.get("y1", 0)),
                float(bbox.get("x2", 1)),
                float(bbox.get("y2", 1))
            )
        elif "xmin" in bbox and "ymin" in bbox:
            return (
                float(bbox.get("xmin", 0)),
                float(bbox.get("ymin", 0)),
                float(bbox.get("xmax", 1)),
                float(bbox.get("ymax", 1))
            )
        elif "ymin" in bbox and "xmin" in bbox:
            return (
                float(bbox.get("xmin", 0)),
                float(bbox.get("ymin", 0)),
                float(bbox.get("xmax", 1)),
                float(bbox.get("ymax", 1))
            )
    elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))

    # Default fallback full page box
    return (0.0, 0.0, 1.0, 1.0)
