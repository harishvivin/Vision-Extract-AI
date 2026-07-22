"""
Segment Anything Model 2 (SAM2) Segmenter Engine.
Generates pixel-accurate segmentation masks for detected bounding boxes,
with seamless fallback to bounding box mask if SAM2 model is unavailable.
"""

import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch

from config import SAM2_MODEL_ID, USE_SAM2, DEVICE

logger = logging.getLogger(__name__)


class SAM2Segmenter:
    """SAM2 Segmenter using Segment Anything Model 2 for fine-grained object mask generation."""

    def __init__(self, model_id: str = SAM2_MODEL_ID):
        """
        Initialize SAM2 Segmenter.

        Args:
            model_id (str): Hugging Face or SAM2 model identifier.
        """
        self.model_id = model_id
        self.device = DEVICE
        self.processor = None
        self.model = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Lazy load SAM2 model and processor."""
        if self._is_loaded or not USE_SAM2:
            return

        logger.info(f"Loading SAM model/segmenter '{self.model_id}' on device '{self.device}'...")
        try:
            from transformers import SamModel, SamProcessor
            # Use SAM/SAM2 HuggingFace interface
            self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
            self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("SAM model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SAM model ('{self.model_id}'): {e}. Will fallback to Bounding Box Cropper.")
            self._is_loaded = False

    def generate_mask(self, image: Image.Image, box: List[float]) -> Tuple[np.ndarray, bool]:
        """
        Generate binary mask for the given bounding box using SAM2 or bounding box fallback.

        Args:
            image (Image.Image): Input photograph image.
            box (List[float]): Bounding box coordinates [x1, y1, x2, y2].

        Returns:
            Tuple[np.ndarray, bool]: Binary mask (H, W) boolean array and boolean flag indicating if SAM2 was used.
        """
        W, H = image.size
        self.load_model()

        if self._is_loaded and self.model is not None and self.processor is not None:
            try:
                # Format box prompt for SAM processor [[[x1, y1, x2, y2]]]
                input_boxes = [[[box[0], box[1], box[2], box[3]]]]
                inputs = self.processor(image, input_boxes=input_boxes, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                masks = self.processor.image_processor.post_process_masks(
                    outputs.pred_masks.cpu(),
                    inputs["original_sizes"].cpu(),
                    inputs["reshaped_input_sizes"].cpu()
                )

                # Get highest score mask
                iou_scores = outputs.iou_scores.cpu().numpy()[0][0]
                best_mask_idx = int(np.argmax(iou_scores))
                mask_np = masks[0][0][best_mask_idx].numpy().astype(bool)

                logger.info(f"SAM Segmentation successful (best IoU score: {iou_scores[best_mask_idx]:.2f}).")
                return mask_np, True

            except Exception as e:
                logger.warning(f"SAM segmentation execution failed: {e}. Falling back to bounding box crop mask.")

        # Fallback to Bounding Box Mask
        mask_np = np.zeros((H, W), dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        mask_np[y1:y2, x1:x2] = True

        logger.info("Using Bounding Box fallback mask.")
        return mask_np, False
