"""FastAPI backend for OpenNoMark."""

import os
import uuid
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from . import __version__

app = FastAPI(title="OpenNoMark", version=__version__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded pipeline singleton
_pipeline = None
UPLOAD_DIR = Path(tempfile.gettempdir()) / "opennomark_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "opennomark_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from .pipeline import WatermarkRemovalPipeline
        _pipeline = WatermarkRemovalPipeline()
    return _pipeline


@app.get("/api/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/api/remove")
async def remove_watermark(files: list[UploadFile] = File(...)):
    """Remove watermarks from uploaded images."""
    pipeline = get_pipeline()
    results = []

    for upload in files:
        if not upload.content_type or not upload.content_type.startswith("image/"):
            results.append({"filename": upload.filename, "error": "Not an image file"})
            continue

        # Save upload
        job_id = uuid.uuid4().hex[:8]
        ext = os.path.splitext(upload.filename or "image.png")[1] or ".png"
        input_path = UPLOAD_DIR / f"{job_id}_input{ext}"
        output_path = OUTPUT_DIR / f"{job_id}_clean{ext}"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(upload.file, f)

        # Process
        _, meta = pipeline.process(str(input_path), str(output_path))

        results.append({
            "filename": upload.filename,
            "job_id": job_id,
            "status": meta["status"],
            "watermarks_found": meta["watermarks_found"],
            "download_url": f"/api/download/{job_id}{ext}" if meta["status"] == "cleaned" else None,
        })

    return {"results": results}


@app.get("/api/download/{filename}")
def download(filename: str):
    """Download a processed image."""
    file_path = OUTPUT_DIR / f"{filename.split('.')[0].replace('_clean', '')}_clean.{filename.split('.')[-1]}"
    # Try to find the file with the job_id
    for f in OUTPUT_DIR.iterdir():
        if filename.split(".")[0] in f.name:
            return FileResponse(f, filename=f"clean_{f.name}")
    raise HTTPException(404, "File not found")


# Serve frontend static files if they exist
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
