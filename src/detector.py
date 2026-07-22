"""
Grounding DINO Zero-Shot Object Detector.
Uses Hugging Face Transformers Grounding DINO with automatic prompt generation,
spatial filtering, and confidence threshold fallback cascades.
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import torch
import numpy as np

from config import (
    GROUNDING_DINO_MODEL_ID,
    GROUNDING_DINO_FALLBACK_MODEL_ID,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
    RETRY_BOX_THRESHOLDS,
    SPATIAL_REGIONS,
    DEVICE
)
from src.utils import calculate_spatial_score
from src.question_parser import ParsedQuestion

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection output structure."""
    box: List[float]  # Pixel coordinates [x1, y1, x2, y2]
    confidence: float
    label: str
    prompt_used: str
    spatial_score: float
    processing_time_ms: float
    attempts_log: List[Dict[str, Any]]


class GroundingDINODetector:
    """Zero-Shot Object Detector using Grounding DINO."""

    def __init__(self, model_id: str = GROUNDING_DINO_MODEL_ID):
        """
        Initialize Grounding DINO detector model and processor.

        Args:
            model_id (str): Hugging Face model repository ID.
        """
        self.model_id = model_id
        self.device = DEVICE
        self.processor = None
        self.model = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Lazy load model and processor to minimize startup memory overhead."""
        if self._is_loaded:
            return

        logger.info(f"Loading Grounding DINO model '{self.model_id}' on device '{self.device}'...")
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
            self.model.eval()
            self._is_loaded = True
            logger.info("Grounding DINO loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_id}: {e}. Trying fallback model...")
            try:
                from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
                self.model_id = GROUNDING_DINO_FALLBACK_MODEL_ID
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
                self.model.eval()
                self._is_loaded = True
                logger.info("Fallback Grounding DINO model loaded successfully.")
            except Exception as ex:
                logger.error(f"Failed to load fallback Grounding DINO model: {ex}")
                self._is_loaded = False

    def detect(self, image: Image.Image, parsed_q: ParsedQuestion) -> DetectionResult:
        """
        Detect the target object using Grounding DINO with spatial position filtering and fallback retries.

        Args:
            image (Image.Image): Input photograph image.
            parsed_q (ParsedQuestion): Parsed question metadata containing object, color, position, prompts.

        Returns:
            DetectionResult: Best bounding box, confidence, and metadata.
        """
        start_time = time.time()
        self.load_model()
        attempts_log = []

        W, H = image.size

        if not self._is_loaded or self.model is None or self.processor is None:
            logger.warning("Grounding DINO model not available. Returning spatial fallback crop box.")
            proc_time = (time.time() - start_time) * 1000
            return DetectionResult(
                box=self._apply_position_fallback(image, parsed_q.position),
                confidence=0.5,
                label=parsed_q.primary_prompt,
                prompt_used="spatial_fallback",
                spatial_score=1.0,
                processing_time_ms=proc_time,
                attempts_log=[{"strategy": "model_unavailable_fallback"}]
            )

        # Retry strategy sequence
        prompt_candidates = [
            ("color_and_object", parsed_q.primary_prompt),
            ("object_only", parsed_q.object_prompt),
            ("detailed_prompt", parsed_q.detailed_prompt)
        ]

        best_candidate: Optional[Tuple[List[float], float, str, float]] = None
        highest_combined_score = -1.0

        for prompt_name, prompt_text in prompt_candidates:
            if not prompt_text:
                continue

            for box_thresh in RETRY_BOX_THRESHOLDS:
                attempt_record = {
                    "prompt_name": prompt_name,
                    "prompt_text": prompt_text,
                    "box_threshold": box_thresh,
                    "detections_found": 0
                }

                boxes, scores, labels = self._run_inference(image, prompt_text, box_thresh)
                attempt_record["detections_found"] = len(boxes)

                if boxes:
                    # Filter and score each bounding box candidate against spatial position requirement
                    for box, score, lbl in zip(boxes, scores, labels):
                        spatial_score = calculate_spatial_score(box, W, H, parsed_q.position)
                        # Combined score = model confidence * spatial positioning match score
                        combined_score = score * (0.6 + 0.4 * spatial_score)

                        if combined_score > highest_combined_score:
                            highest_combined_score = combined_score
                            best_candidate = (box, score, prompt_text, spatial_score)

                    attempt_record["best_combined_score"] = float(highest_combined_score)
                    attempts_log.append(attempt_record)

                    # If confidence is strong (>0.35) and spatial match is good, accept early
                    if highest_combined_score >= 0.35:
                        break

            if best_candidate and highest_combined_score >= 0.35:
                break

        proc_time = (time.time() - start_time) * 1000

        if best_candidate:
            box, conf, prompt_used, s_score = best_candidate
            logger.info(f"Detection Success -> Box: {box}, Conf: {conf:.2f}, Prompt: '{prompt_used}', SpatialScore: {s_score:.2f}")
            return DetectionResult(
                box=box,
                confidence=float(conf),
                label=parsed_q.primary_prompt,
                prompt_used=prompt_used,
                spatial_score=float(s_score),
                processing_time_ms=proc_time,
                attempts_log=attempts_log
            )
        else:
            logger.warning(f"No model detection met threshold for '{parsed_q.primary_prompt}'. Applying spatial position heuristic box.")
            heuristic_box = self._apply_position_fallback(image, parsed_q.position)
            return DetectionResult(
                box=heuristic_box,
                confidence=0.3,
                label=parsed_q.primary_prompt,
                prompt_used="position_heuristic_fallback",
                spatial_score=0.8,
                processing_time_ms=proc_time,
                attempts_log=attempts_log
            )

    def _run_inference(self, image: Image.Image, prompt: str, box_threshold: float) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Execute forward pass of Grounding DINO on image and text prompt.

        Args:
            image (Image.Image): Input image.
            prompt (str): Detection text prompt.
            box_threshold (float): Confidence threshold for box selection.

        Returns:
            Tuple[List[List[float]], List[float], List[str]]: Bounding boxes, scores, and labels.
        """
        # Grounding DINO expects lowercased prompt ending with a period
        formatted_prompt = prompt.lower().strip()
        if not formatted_prompt.endswith('.'):
            formatted_prompt += '.'

        try:
            inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process outputs
            target_sizes = [image.size[::-1]]  # (height, width)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs.input_ids,
                threshold=box_threshold,
                text_threshold=DEFAULT_TEXT_THRESHOLD,
                target_sizes=target_sizes
            )[0]

            boxes = results["boxes"].cpu().numpy().tolist()
            scores = results["scores"].cpu().numpy().tolist()
            labels = results.get("labels", [prompt] * len(boxes))

            return boxes, scores, labels

        except Exception as e:
            logger.warning(f"Inference exception for prompt '{prompt}' (thresh={box_threshold}): {e}")
            return [], [], []

    def _apply_position_fallback(self, image: Image.Image, position: Optional[str]) -> List[float]:
        """
        Generate heuristic bounding box based on requested position when detection yields no box.

        Args:
            image (Image.Image): Input image.
            position (Optional[str]): Position descriptor e.g. 'bottom-left'.

        Returns:
            List[float]: Bounding box [x1, y1, x2, y2].
        """
        W, H = image.size
        if not position or position.lower() not in SPATIAL_REGIONS:
            return [float(W * 0.15), float(H * 0.15), float(W * 0.85), float(H * 0.85)]

        norm_box = SPATIAL_REGIONS[position.lower()]
        return [
            float(norm_box[0] * W),
            float(norm_box[1] * H),
            float(norm_box[2] * W),
            float(norm_box[3] * H)
        ]
