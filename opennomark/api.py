"""FastAPI backend for OpenNoMark."""

import os
import re
import uuid
import shutil
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.background import BackgroundTask

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


class BatchDownloadItem(BaseModel):
    job_id: str
    filename: str


class BatchDownloadRequest(BaseModel):
    items: list[BatchDownloadItem]


def _output_for_job(job_id: str) -> Path | None:
    """Resolve a generated output without accepting arbitrary paths."""
    if not re.fullmatch(r"[0-9a-f]{8}", job_id):
        return None
    return next(OUTPUT_DIR.glob(f"{job_id}_clean.*"), None)


def _unique_archive_name(filename: str, suffix: str, used: set[str]) -> str:
    """Build a readable, collision-free filename for a batch archive."""
    safe_name = Path(filename.replace("\\", "/")).name or f"result{suffix}"
    source = Path(safe_name)
    ext = suffix
    stem = source.stem or "result"
    candidate = f"clean_{stem}{ext}"
    index = 2
    while candidate in used:
        candidate = f"clean_{stem}_{index}{ext}"
        index += 1
    used.add(candidate)
    return candidate


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

        try:
            _, meta = pipeline.process(str(input_path), str(output_path))
            if meta["status"] == "partial":
                results.append({
                    "filename": upload.filename,
                    "status": "error",
                    "watermarks_found": meta["watermarks_found"],
                    "download_url": None,
                    "error": "Residual watermark evidence remained after validation",
                })
                continue
            if meta["status"] == "no_watermark":
                # An unchanged image is still a valid batch result. Keeping a
                # copy in the output directory makes single and ZIP downloads
                # consistent for every successfully processed item.
                shutil.copyfile(input_path, output_path)

            results.append({
                "filename": upload.filename,
                "job_id": job_id,
                "status": meta["status"],
                "watermarks_found": meta["watermarks_found"],
                "download_url": f"/api/download/{job_id}{ext}",
            })
        except Exception as exc:
            results.append({
                "filename": upload.filename,
                "status": "error",
                "watermarks_found": 0,
                "download_url": None,
                "error": str(exc),
            })

    return {"results": results}


@app.get("/api/download/{filename}")
def download(filename: str):
    """Download a processed image."""
    job_id = Path(filename).stem.replace("_clean", "")
    file_path = _output_for_job(job_id)
    if file_path:
        return FileResponse(file_path, filename=f"clean_{file_path.name}")
    raise HTTPException(404, "File not found")


@app.post("/api/download-batch")
def download_batch(request: BatchDownloadRequest):
    """Bundle all available processed outputs into one ZIP download."""
    if not request.items:
        raise HTTPException(400, "No files requested")

    archive_file = tempfile.NamedTemporaryFile(
        prefix="opennomark_batch_", suffix=".zip", delete=False
    )
    archive_path = Path(archive_file.name)
    archive_file.close()
    used_names: set[str] = set()
    included = 0
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for item in request.items:
                file_path = _output_for_job(item.job_id)
                if not file_path:
                    continue
                archive_name = _unique_archive_name(item.filename, file_path.suffix, used_names)
                bundle.write(file_path, arcname=archive_name)
                included += 1
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise

    if not included:
        archive_path.unlink(missing_ok=True)
        raise HTTPException(404, "No processed files found")

    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename="opennomark-results.zip",
        background=BackgroundTask(archive_path.unlink, missing_ok=True),
    )


# Serve frontend static files if they exist
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
