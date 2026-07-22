"""
Utility Functions for Spatial Reasoning, Visualization, Logging, and Packaging.
"""

import os
import json
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import SPATIAL_REGIONS, OUTPUTS_DIR, LOGS_DIR

logger = logging.getLogger(__name__)


def calculate_spatial_score(box: List[float], img_width: int, img_height: int, position_key: Optional[str]) -> float:
    """
    Calculate spatial score (0.0 to 1.0) indicating how well a bounding box matches
    the specified spatial position descriptor (e.g. 'bottom-left', 'top-centre').

    Args:
        box (List[float]): Bounding box [x_min, y_min, x_max, y_max] in pixels.
        img_width (int): Image width.
        img_height (int): Image height.
        position_key (Optional[str]): Position description string.

    Returns:
        float: Spatial alignment score (1.0 = perfect match, 0.0 = outside target zone).
    """
    if not position_key or position_key.lower() not in SPATIAL_REGIONS:
        return 1.0  # Neutral score if position not specified or unknown

    target_region = SPATIAL_REGIONS[position_key.lower()]
    norm_x1, norm_y1, norm_x2, norm_y2 = target_region

    # Normalize box coordinates (0 to 1)
    bx1, by1, bx2, by2 = box[0] / img_width, box[1] / img_height, box[2] / img_width, box[3] / img_height
    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    # Check center overlap in target region
    in_x = norm_x1 <= bcx <= norm_x2
    in_y = norm_y1 <= bcy <= norm_y2

    if in_x and in_y:
        # Distance from region center
        tcx = (norm_x1 + norm_x2) / 2.0
        tcy = (norm_y1 + norm_y2) / 2.0
        dist = np.sqrt((bcx - tcx)**2 + (bcy - tcy)**2)
        score = max(0.5, 1.0 - dist)
        return float(score)
    else:
        # Penalty if outside target region
        return 0.1


def draw_detection_overlay(
    image: Image.Image,
    box: List[float],
    label: str,
    confidence: float,
    mask: Optional[np.ndarray] = None
) -> Image.Image:
    """
    Draw bounding box, label, confidence, and mask overlay on an image for visualization.

    Args:
        image (Image.Image): Input PIL Image.
        box (List[float]): Bounding box [x1, y1, x2, y2].
        label (str): Class or prompt label.
        confidence (float): Confidence score.
        mask (Optional[np.ndarray]): Binary segmentation mask.

    Returns:
        Image.Image: Image with visual overlay applied.
    """
    canvas = image.copy().convert("RGBA")
    W, H = canvas.size

    # Apply semi-transparent mask overlay if provided
    if mask is not None:
        mask_np = np.array(mask, dtype=bool)
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        # Lime green tint for mask
        overlay[mask_np] = [0, 255, 128, 100]
        overlay_img = Image.fromarray(overlay, mode="RGBA")
        canvas = Image.alpha_composite(canvas, overlay_img)

    draw = ImageDraw.Draw(canvas)
    x1, y1, x2, y2 = box

    # Draw primary bounding box line
    draw.rectangle([x1, y1, x2, y2], outline="#00FF88", width=4)

    # Draw text banner
    text = f"{label} ({confidence:.2f})"
    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()

    bbox_text = draw.textbbox((x1, max(0, y1 - 25)), text, font=font)
    draw.rectangle(bbox_text, fill="#00FF88")
    draw.text((x1 + 4, max(0, y1 - 25) + 2), text, fill="black", font=font)

    return canvas.convert("RGB")


def create_output_zip(output_dir: Path = OUTPUTS_DIR) -> Path:
    """
    Create a ZIP archive containing all extracted output PNG files.

    Args:
        output_dir (Path): Output directory containing saved images.

    Returns:
        Path: Path to created zip file.
    """
    zip_path = output_dir / "all_extracted_objects.zip"
    logger.info(f"Creating zip file at: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in output_dir.glob("*.png"):
            zipf.write(file_path, arcname=file_path.name)

    logger.info(f"Successfully packaged {len(list(output_dir.glob('*.png')))} images into ZIP.")
    return zip_path


def log_detection_data(data: Dict[str, Any], log_file: Path = LOGS_DIR / "detections.json") -> None:
    """
    Append detection telemetry record to JSON log file.

    Args:
        data (Dict[str, Any]): Log dictionary for single page processing.
        log_file (Path): Target log path.
    """
    logs = []
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except Exception:
            logs = []

    logs.append(data)

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
