"""
Image Cropper Engine.
Crops target detected objects cleanly using segmentation masks or bounding boxes
and saves output PNG images.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image

from config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class ImageCropper:
    """Cropper class for extracting segmented or bounding-box regions from images."""

    def __init__(self, output_dir: Path = OUTPUTS_DIR):
        """
        Initialize ImageCropper.

        Args:
            output_dir (Path): Output directory for saving cropped images.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def crop_and_save(
        self,
        image: Image.Image,
        box: List[float],
        mask: Optional[np.ndarray],
        filename: str,
        transparent_bg: bool = False
    ) -> Tuple[Image.Image, Path]:
        """
        Crop the detected region and save to file with specified filename.

        Args:
            image (Image.Image): Input photograph image.
            box (List[float]): Bounding box [x1, y1, x2, y2].
            mask (Optional[np.ndarray]): Binary mask array (H, W).
            filename (str): Target filename e.g., '01_yellow_tulips.png'.
            transparent_bg (bool): If True, isolates object on transparent PNG background.

        Returns:
            Tuple[Image.Image, Path]: Cropped PIL Image and saved file path.
        """
        W, H = image.size
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, max(x1 + 1, x2)), min(H, max(y1 + 1, y2))

        if transparent_bg and mask is not None:
            # Create RGBA image with transparent background outside mask
            img_np = np.array(image.convert("RGBA"))
            alpha_mask = (mask * 255).astype(np.uint8)
            img_np[:, :, 3] = alpha_mask
            
            # Crop to bounding box
            cropped_np = img_np[y1:y2, x1:x2]
            cropped_img = Image.fromarray(cropped_np, mode="RGBA")
        else:
            # Clean RGB bounding box crop
            cropped_img = image.crop((x1, y1, x2, y2))

        save_path = self.output_dir / filename
        cropped_img.save(save_path, format="PNG")

        logger.info(f"Cropped image saved: '{save_path}' (Dimensions: {cropped_img.width}x{cropped_img.height} px)")
        return cropped_img, save_path
