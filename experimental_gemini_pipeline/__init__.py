"""
Experimental Gemini-Only PDF Localization & Cropping Pipeline Package.
"""

from .config import GEMINI_API_KEY_PRIMARY, GEMINI_API_KEY_FALLBACK, OUTPUTS_DIR
from .prompt_builder import build_prompt
from .gemini_client import ExperimentalGeminiClient
from .coordinate_cropper import crop_pdf_region
from .main import ExperimentalPipeline

__all__ = [
    "ExperimentalGeminiClient",
    "ExperimentalPipeline",
    "crop_pdf_region",
    "build_prompt",
    "GEMINI_API_KEY_PRIMARY",
    "GEMINI_API_KEY_FALLBACK",
    "OUTPUTS_DIR"
]
