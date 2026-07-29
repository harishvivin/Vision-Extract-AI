# 🧪 Experimental Gemini-Only PDF Localization & Cropping Pipeline

A completely separate, standalone experimental pipeline for visual PDF document localization and evidence cropping using **Google Gemini API** (Flash-Lite model) and **PyMuPDF (`fitz`)**.

> [!IMPORTANT]
> This pipeline is completely isolated in `experimental_gemini_pipeline/`. It does **NOT** modify or depend on the main project, FastAPI backend, React frontend, or existing object detectors.

---

## 🎯 Architectural Overview

```
┌─────────────────────────────────┐
│     User Uploads Medical PDF    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  ExperimentalGeminiClient       │ ◄── Primary / Fallback Key Failover
│  (Gemini Flash-Lite, Temp 0.0)  │
└────────────────┬────────────────┘
                 │
                 ▼  Returns JSON: { found: true, page: X, bounding_box: {x1, y1, x2, y2} }
┌─────────────────────────────────┐
│  CoordinateCropper              │
│  (Pure PyMuPDF clip rendering)  │ ◄── Zero OpenCV, Zero OCR, Zero Screenshots
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  High-Res PNG Evidence Crop     │ (outputs_experimental/crop_01_patient_name.png)
└─────────────────────────────────┘
```

---

## 🚀 Key Features & Constraints

1. **Zero Object Detectors**: No YOLO, Grounding DINO, SAM, OpenCV, or pretrained detection models.
2. **Pure Gemini Vision + PyMuPDF**: Gemini localizes the exact bounding box and page number; PyMuPDF renders the high-DPI clipped vector sub-region directly into a PNG file.
3. **Dual API Key Transparent Failover**:
   - `GEMINI_API_KEY_PRIMARY`
   - `GEMINI_API_KEY_FALLBACK`
   - Automatically and transparently retries with `GEMINI_API_KEY_FALLBACK` upon any API exception (rate limit, quota exceeded, timeout, status code error).
4. **Strict JSON Schema**: Uses Python f-strings prompt builder enforcing strict output structure.

---

## 📁 File Structure

```
experimental_gemini_pipeline/
├── config.py             # Environment configuration & dual API key definitions
├── prompt_builder.py     # Python f-string prompt generator
├── gemini_client.py       # Dual API key Gemini client & PDF upload engine
├── coordinate_cropper.py # Lossless PyMuPDF vector region cropper
├── main.py               # Master orchestration script & test suite CLI
├── __init__.py           # Python package exports
└── README.md             # Documentation
```

---

## 💻 How to Run

### 1. Set Environment Variables (Optional)
In your root `.env` file:
```env
GEMINI_API_KEY_PRIMARY=your_primary_gemini_api_key
GEMINI_API_KEY_FALLBACK=your_fallback_gemini_api_key
```

### 2. Run Main Pipeline Test Script
```bash
# Run against default sample PDF
py -m experimental_gemini_pipeline.main

# Or specify a custom PDF document
py -m experimental_gemini_pipeline.main "path/to/medical_report.pdf"
```
