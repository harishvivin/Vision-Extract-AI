"""
Configuration module for the experimental Gemini-only PDF localization pipeline.
Handles dual API key authentication, model defaults, and output path configuration.
"""

import os
from pathlib import Path

# Base Paths
EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
OUTPUTS_DIR = EXPERIMENT_DIR / "outputs_experimental"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def _load_env():
    """Load environment variables from project root .env if available."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

_load_env()

# Dual API Key Environment Variables
GEMINI_API_KEY_PRIMARY = (
    os.environ.get("GEMINI_API_KEY_PRIMARY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)

GEMINI_API_KEY_FALLBACK = (
    os.environ.get("GEMINI_API_KEY_FALLBACK")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)

# Preferred Gemini Models in order of preference (Flash-Lite focus)
GEMINI_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-8b",
    "gemini-2.0-flash",
    "gemini-1.5-flash"
]

DEFAULT_TEMPERATURE = 0.0
DEFAULT_DPI = 300
