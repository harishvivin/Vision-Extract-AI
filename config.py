"""
Configuration Settings for Vision Extract AI Application.
Centralized settings management using Python Dataclasses / Pydantic.
"""

from pathlib import Path
from typing import List, Dict
import torch

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Hardware Device Selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PDF Extraction Settings
PDF_DPI = 300  # High resolution for optimal vision model performance

# Grounding DINO Detection Settings
# Hugging Face Model ID for Grounding DINO
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
# Alternative fallback model ID
GROUNDING_DINO_FALLBACK_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Confidence Thresholds
DEFAULT_BOX_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.25
RETRY_BOX_THRESHOLDS = [0.25, 0.20, 0.15, 0.10]

# SAM 2 (Segment Anything 2) Settings
SAM2_MODEL_ID = "facebook/sam2-hiera-tiny"
USE_SAM2 = True  # Enable SAM2 segmentation with bounding box fallback if unavailable

# Spatial Region Keywords Mapping
SPATIAL_REGIONS: Dict[str, List[float]] = {
    # [min_x, min_y, max_x, max_y] normalized coordinates (0.0 to 1.0)
    "bottom-left": [0.0, 0.45, 0.55, 1.0],
    "bottom-right": [0.45, 0.45, 1.0, 1.0],
    "top-left": [0.0, 0.0, 0.55, 0.55],
    "top-right": [0.45, 0.0, 1.0, 0.55],
    "top-centre": [0.2, 0.0, 0.8, 0.6],
    "top-center": [0.2, 0.0, 0.8, 0.6],
    "front-right": [0.35, 0.2, 1.0, 1.0],
    "front-centre": [0.15, 0.2, 0.85, 1.0],
    "front-center": [0.15, 0.2, 0.85, 1.0],
    "right side": [0.4, 0.0, 1.0, 1.0],
    "right": [0.4, 0.0, 1.0, 1.0],
    "left side": [0.0, 0.0, 0.6, 1.0],
    "left": [0.0, 0.0, 0.6, 1.0],
    "centre": [0.15, 0.15, 0.85, 0.85],
    "center": [0.15, 0.15, 0.85, 0.85],
    "front": [0.0, 0.3, 1.0, 1.0],
    "foreground": [0.0, 0.3, 1.0, 1.0],
    "middle": [0.2, 0.1, 0.8, 0.9]
}

# Logging Settings
LOG_FILE = LOGS_DIR / "pipeline.log"
DETECTION_LOG_FILE = LOGS_DIR / "detections.json"
