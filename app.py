"""
FastAPI Backend Server for Vision Extract AI Application.
Provides REST API endpoints for PDF processing, image previewing, log retrieval, and downloading output ZIPs.
"""

import os
import io
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import OUTPUTS_DIR, LOGS_DIR, BASE_DIR
from src.pipeline import ExtractionPipeline
from src.qa_engine import DocumentQAEngine
from pydantic import BaseModel

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGS_DIR / "app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app")

# Initialize FastAPI App
app = FastAPI(
    title="Vision Extract AI",
    description="Automated AI PDF Object Extraction API using Grounding DINO, SAM2, and PyMuPDF",
    version="1.0.0"
)

# Enable CORS for Frontend Development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Pipeline & QA Instances
pipeline = ExtractionPipeline(output_dir=OUTPUTS_DIR, log_dir=LOGS_DIR)
qa_engine = DocumentQAEngine(outputs_dir=OUTPUTS_DIR)

# Directory for storing page visualization overlay previews
PREVIEWS_DIR = OUTPUTS_DIR / "previews"
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

class QAQueryRequest(BaseModel):
    question: str


@app.get("/api/health")
def health_check() -> Dict[str, str]:
    """System health check endpoint."""
    return {
        "status": "healthy",
        "device": pipeline.detector.device,
        "detector_model": pipeline.detector.model_id,
        "sam2_enabled": str(pipeline.segmenter.model_id)
    }


@app.post("/api/process")
async def process_pdf(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a PDF document and process all pages through the AI pipeline.

    Args:
        file (UploadFile): Uploaded PDF file stream.

    Returns:
        Dict[str, Any]: Structured results for all processed pages.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a valid PDF document.")

    # Save uploaded file temporarily
    temp_pdf_path = BASE_DIR / f"temp_{file.filename}"
    try:
        with open(temp_pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Received PDF upload: {file.filename}. Running pipeline...")
        results = pipeline.run(temp_pdf_path)

        page_data_list = []
        for res in results:
            # Save visual overlay preview image
            overlay_filename = f"preview_page_{res.page_number}.png"
            overlay_path = PREVIEWS_DIR / overlay_filename
            res.overlay_image.save(overlay_path, format="PNG")

            page_data_list.append({
                "page_number": res.page_number,
                "raw_question": res.raw_question,
                "parsed_question": res.parsed_question,
                "detection_prompt": res.detection_prompt,
                "confidence": round(res.confidence, 4),
                "bounding_box": [round(b, 2) for b in res.bounding_box],
                "spatial_score": round(res.spatial_score, 4),
                "sam2_used": res.sam2_used,
                "processing_time_ms": round(res.processing_time_ms, 2),
                "output_filename": res.output_filename,
                "output_url": f"/api/outputs/{res.output_filename}",
                "preview_url": f"/api/previews/{overlay_filename}"
            })

        return {
            "success": True,
            "filename": file.filename,
            "total_pages": len(results),
            "pages": page_data_list,
            "download_zip_url": "/api/download-all"
        }

    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")

    finally:
        if temp_pdf_path.exists():
            try:
                temp_pdf_path.unlink()
            except Exception:
                pass


@app.get("/api/outputs/{filename}")
def get_output_file(filename: str):
    """Serve individual cropped output PNG images."""
    file_path = OUTPUTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Output file '{filename}' not found.")
    return FileResponse(file_path, media_type="image/png")


@app.get("/api/previews/{filename}")
def get_preview_file(filename: str):
    """Serve visualization overlay preview PNG images."""
    file_path = PREVIEWS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Preview file '{filename}' not found.")
    return FileResponse(file_path, media_type="image/png")


@app.get("/api/download-all")
def download_all_zip():
    """Download ZIP package containing all 10 cropped object output PNGs."""
    zip_path = OUTPUTS_DIR / "all_extracted_objects.zip"
    if not zip_path.exists():
        from src.utils import create_output_zip
        zip_path = create_output_zip(OUTPUTS_DIR)

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename="all_extracted_objects.zip"
    )


@app.get("/api/logs")
def get_logs() -> List[Dict[str, Any]]:
    """Retrieve detailed detection logs."""
    log_json = LOGS_DIR / "detections.json"
    if not log_json.exists():
        return []
    import json
    try:
        with open(log_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


@app.get("/api/results")
def get_existing_results() -> Dict[str, Any]:
    """Retrieve existing processed page results from telemetry log if available."""
    log_json = LOGS_DIR / "detections.json"
    if not log_json.exists():
        return {"success": False, "pages": []}
    import json
    try:
        with open(log_json, "r", encoding="utf-8") as f:
            logs = json.load(f)

        pages = []
        for item in logs:
            page_num = item.get("page_number", 1)
            overlay_filename = f"preview_page_{page_num}.png"
            pages.append({
                "page_number": page_num,
                "raw_question": item.get("raw_question", ""),
                "parsed_question": item.get("parsed_question", {}),
                "detection_prompt": item.get("detection_prompt", ""),
                "confidence": round(item.get("confidence", 0), 4),
                "bounding_box": [round(b, 2) for b in item.get("bounding_box", [0, 0, 0, 0])],
                "spatial_score": round(item.get("spatial_score", 0), 4),
                "sam2_used": item.get("sam2_used", False),
                "processing_time_ms": round(item.get("processing_time_ms", 0), 2),
                "output_filename": item.get("output_filename", ""),
                "output_url": f"/api/outputs/{item.get('output_filename', '')}",
                "preview_url": f"/api/previews/{overlay_filename}"
            })
        return {
            "success": True,
            "total_pages": len(pages),
            "pages": pages,
            "download_zip_url": "/api/download-all"
        }
    except Exception:
        return {"success": False, "pages": []}


@app.post("/api/qa/ask")
def ask_question(body: QAQueryRequest) -> Dict[str, Any]:
    """Process natural language question about document and return answer with screenshot details."""
    q_result = qa_engine.ask(body.question)
    return {
        "success": True,
        "question": q_result.question,
        "answer": q_result.answer,
        "page_number": q_result.page_number,
        "secondary_page_number": q_result.secondary_page_number,
        "confidence": q_result.confidence,
        "section_title": q_result.section_title,
        "bounding_box": q_result.bounding_box,
        "snippet_filename": q_result.snippet_filename,
        "snippet_url": f"/api/qa/snippets/{q_result.snippet_filename}",
        "preview_url": f"/api/previews/preview_page_{q_result.page_number}.png"
    }


@app.get("/api/qa/sample-questions")
def get_sample_questions() -> List[Dict[str, Any]]:
    """Return pre-configured sample questions."""
    return qa_engine.get_sample_questions()


@app.get("/api/qa/snippets/{filename}")
def get_qa_snippet(filename: str):
    """Serve cropped QA evidence screenshot images."""
    file_path = OUTPUTS_DIR / "qa_snippets" / filename
    if not file_path.exists():
        # Fallback to preview image if snippet not available yet
        fallback_path = PREVIEWS_DIR / "preview_page_10.png"
        if fallback_path.exists():
            return FileResponse(fallback_path, media_type="image/png")
        raise HTTPException(status_code=404, detail=f"Snippet '{filename}' not found.")
    return FileResponse(file_path, media_type="image/png")



# Serve React static build if available
frontend_dist = BASE_DIR / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

