# Vision Extract AI - Automated PDF Object Extraction Pipeline

An enterprise-grade, production-ready AI application designed to automatically parse multi-page PDFs, extract photographs, parse natural language object extraction questions using NLP, detect requested target objects zero-shot using **Grounding DINO**, segment precise pixel-level object masks using **SAM 2 (Segment Anything 2)**, crop target objects, and save them with requested filenames.

---

## 🌟 Key Features

- 📄 **PyMuPDF Document Reader**: Renders high-DPI PDF pages and extracts embedded images and natural language questions.
- 🧠 **NLP Question Parser**: Uses spaCy & rule-based regex parsing to extract object name, color attribute, spatial position, and target filename.
- 🎯 **Grounding DINO Zero-Shot Detection**: Prompts vision-language model automatically with fallback cascades (e.g., `color + object` $\rightarrow$ `object only` $\rightarrow$ confidence thresholds).
- 📍 **Spatial Position Ranker**: Uses normalized coordinate geometry to disambiguate target objects specified by positional hints (e.g., `bottom-left`, `top-centre`, `front-right`).
- ✂️ **SAM 2 Instance Masking**: Generates pixel-precise masks for object boundaries, with seamless fallback to rectangular bounding box cropping if SAM 2 model is unavailable.
- ⚡ **FastAPI REST Server**: Modular backend API providing `/api/process`, `/api/download-all`, `/api/logs`, and static asset serving.
- 🎨 **Modern React + Tailwind Frontend**: Drag-and-drop PDF uploader, real-time progress tracker, visual bounding box/mask preview toggles, dark mode, and single-click ZIP downloader.
- 📊 **Telemetry & Logging**: Detailed JSON logging of detection prompts, spatial scores, confidence levels, bounding box coordinates, and processing time per page.

---

## 🏗️ Architecture & Workflow

```
[ INPUT PDF ] 
      │
      ▼
┌─────────────────────────┐
│  PyMuPDF PDFReader      │ ──► Render Page Image & Extract Question Text
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  QuestionParser (NLP)   │ ──► { object, color, position, filename }
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│ Grounding DINO Detector │ ──► Zero-Shot Bounding Box & Spatial Ranker
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   SAM 2 Segmenter       │ ──► Precise Pixel Mask (or Box Fallback)
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│   ImageCropper Engine   │ ──► Crop & Save to outputs/01_yellow_tulips.png
└─────────────────────────┘
```

---

## 📁 Folder Structure

```
Vision Extract AI/
├── app.py                     # FastAPI REST API server & endpoints
├── config.py                  # Centralized configuration settings & thresholds
├── requirements.txt           # Python dependencies
├── test_pipeline.py           # Integration test script for verifying output files
├── README.md                  # Comprehensive project documentation
├── INPUT_images_and_questions.pdf # Assignment input PDF
├── src/                       # Core Python AI Source Code
│   ├── __init__.py
│   ├── pdf_reader.py          # PyMuPDF document rendering & text extraction
│   ├── question_parser.py     # NLP question parser (object, color, position, filename)
│   ├── detector.py            # Grounding DINO zero-shot detector with fallback retry logic
│   ├── segmenter.py           # SAM 2 instance segmenter with bounding box fallback
│   ├── cropper.py             # Precise pixel/box cropper and image exporter
│   ├── pipeline.py            # Master pipeline orchestrator
│   └── utils.py               # Spatial scoring, visualizer, zip archiver, JSON logger
├── frontend/                  # React 18 + Tailwind CSS Modern Web UI
│   ├── src/
│   │   ├── components/        # Navbar, UploadZone, ProgressBar, PageCard, LogsModal
│   │   ├── App.jsx            # Main React App Component
│   │   ├── main.jsx
│   │   └── index.css          # Tailwind CSS styles
│   ├── vite.config.js         # Vite configuration with API proxy
│   └── package.json
├── models/                    # Model cache directory
├── outputs/                   # Extracted cropped PNG files & ZIP archive
└── logs/                      # Telemetry logs (detections.json, pipeline.log)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js v18+ & npm
- PyTorch (CPU or CUDA GPU)

### 1. Python Environment Setup
```bash
# Clone or navigate to the project root directory
cd "Vision Extract AI"

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Build static bundle (optional, for backend static serving)
npm run build
```

---

## 💻 How to Run

### Method 1: CLI Integration Test (Instant Pipeline Execution)
To run the automated pipeline on `INPUT_images_and_questions.pdf` directly from the command line:
```bash
python test_pipeline.py
```
This will process all 10 pages, populate `outputs/`, create `all_extracted_objects.zip`, and log telemetry into `logs/detections.json`.

### Method 2: Launch Full Application (FastAPI + React UI)
1. **Start the FastAPI Backend Server**:
```bash
python app.py
```
*The API server will run at `http://localhost:8000`.*

2. **Start the React Frontend Dev Server** (in a separate terminal):
```bash
cd frontend
npm run dev
```
*Open `http://localhost:3000` in your web browser to access the UI.*

---

## 📋 Parsed Question Examples

| Page | Question Excerpt | Parsed Object | Parsed Color | Parsed Position | Output Filename |
|:---|:---|:---|:---|:---|:---|
| Page 1 | "Crop out only the bunch of YELLOW tulips (bottom-left)..." | tulips | yellow | bottom-left | `01_yellow_tulips.png` |
| Page 2 | "Crop out only the cluster of RED tulips (top-centre)..." | tulips | red | top-centre | `02_red_tulips.png` |
| Page 3 | "Crop out only the SILVER car on the front-right..." | car | silver | front-right | `03_silver_car.png` |
| Page 4 | "Crop out only the GREEN-and-yellow patterned balloon..." | balloon | green | right side | `04_green_balloon.png` |
| Page 5 | "Crop out only the pile of green PEARS (centre)..." | pears | green | centre | `05_pears.png` |
| Page 6 | "Crop out only the MIDDLE hanging braid of peppers..." | peppers | N/A | middle | `06_middle_braid.png` |
| Page 7 | "Crop out only the large SHEEP standing in the front-centre..." | sheep | N/A | front-centre | `07_front_sheep.png` |
| Page 8 | "Crop out only the DUCK with the green head (front)..." | duck | green | front | `08_green_duck.png` |
| Page 9 | "Crop out only the GREEN (pistachio) macaron..." | macaron | green | N/A | `09_green_macaron.png` |
| Page 10 | "Crop out only the pink FLAMINGO in the foreground..." | flamingo | pink | foreground | `10_flamingo.png` |

---

## ⚙️ Retry & Fallback Strategies

If object detection fails or confidence falls below threshold:
1. **Prompt Generalization**: System retries with `color + object`, then `object only`.
2. **Confidence Threshold Decay**: Threshold decays dynamically from `0.25` down to `0.10`.
3. **Spatial Position Fallback**: If zero model bounding boxes match, a spatial position heuristic bounding box is generated based on the requested target region (e.g. `bottom-left`).
4. **SAM 2 Fallback**: If SAM 2 model execution or GPU/CPU allocation fails, the pipeline falls back gracefully to cropping the bounding box directly.

---

## 🔍 Assumptions, Limitations & Future Improvements

### Assumptions
- Each page contains a single primary photograph and a printed question.
- The requested target filename format ends with `.png`.

### Limitations
- CPU inference for Grounding DINO + SAM 2 takes a few seconds per page (CUDA acceleration recommended for sub-second speeds).

### Future Improvements
- **SAM 2 Video Point Prompts**: Multi-point interactive click refinement in the React frontend.
- **Vision LLM Integration**: Incorporating Qwen-VL or LLaVA for complex multi-modal spatial reasoning.
