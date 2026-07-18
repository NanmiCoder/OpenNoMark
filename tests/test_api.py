"""Tests for FastAPI backend."""

import asyncio
import os
import io
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import UploadFile
from PIL import Image
from starlette.datastructures import Headers


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from opennomark.api import app
    return TestClient(app)


class TestAPI:
    """Test FastAPI endpoints."""

    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert 1 <= data["max_concurrency"] <= 4

    def test_remove_single(self, client, sample_image):
        with open(sample_image, "rb") as f:
            resp = client.post(
                "/api/remove",
                files=[("files", ("test.png", f, "image/png"))],
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["status"] in ("cleaned", "no_watermark")
        assert "watermarks_found" in r
        assert r["download_url"] is not None
        assert client.get(r["download_url"]).status_code == 200

    def test_remove_multiple(self, client, sample_images_dir):
        files = []
        for fname in os.listdir(sample_images_dir):
            path = os.path.join(sample_images_dir, fname)
            files.append(("files", (fname, open(path, "rb"), "image/png")))

        resp = client.post("/api/remove", files=files)
        # Close file handles
        for _, (_, fh, _) in files:
            fh.close()

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == len(files)

    def test_remove_multiple_uses_bounded_concurrency_and_preserves_order(
        self, sample_image, monkeypatch
    ):
        import opennomark.api as api

        class TrackingPipeline:
            def __init__(self):
                self.lock = threading.Lock()
                self.active = 0
                self.max_active = 0
                self.calls = 0

            def process(self, input_path, output_path):
                with self.lock:
                    call_index = self.calls
                    self.calls += 1
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                try:
                    # The second task finishes first, proving response order does
                    # not depend on completion order.
                    time.sleep(0.08 if call_index == 0 else 0.02)
                    image = Image.open(input_path).convert("RGB")
                    image.save(output_path)
                    return image, {
                        "status": "cleaned",
                        "watermarks_found": 1,
                    }
                finally:
                    with self.lock:
                        self.active -= 1

        pipeline = TrackingPipeline()
        monkeypatch.setattr(api, "_pipeline", pipeline)
        monkeypatch.setattr(api, "_processing_slots", asyncio.Semaphore(2))
        image_bytes = Path(sample_image).read_bytes()
        uploads = [
            UploadFile(
                io.BytesIO(image_bytes),
                filename=f"image_{index}.png",
                headers=Headers({"content-type": "image/png"}),
            )
            for index in range(3)
        ]

        response = asyncio.run(api.remove_watermark(uploads))

        try:
            assert pipeline.max_active == 2
            assert [item["filename"] for item in response["results"]] == [
                "image_0.png",
                "image_1.png",
                "image_2.png",
            ]
        finally:
            for upload in uploads:
                upload.file.close()
            for item in response["results"]:
                output = api._output_for_job(item["job_id"])
                if output:
                    output.unlink(missing_ok=True)

    def test_pipeline_initialization_is_shared_across_threads(self, monkeypatch):
        import opennomark.api as api

        sentinel = object()
        calls = 0
        calls_lock = threading.Lock()

        def create_pipeline():
            nonlocal calls
            with calls_lock:
                calls += 1
            time.sleep(0.03)
            return sentinel

        monkeypatch.setattr(api, "_pipeline", None)
        monkeypatch.setattr(api, "_create_pipeline", create_pipeline)
        with ThreadPoolExecutor(max_workers=4) as executor:
            pipelines = list(executor.map(lambda _: api.get_pipeline(), range(4)))

        assert calls == 1
        assert all(pipeline is sentinel for pipeline in pipelines)

    def test_partial_validation_is_exposed_as_retryable_error(
        self, client, sample_image, monkeypatch
    ):
        import opennomark.api as api

        class PartialPipeline:
            def process(self, input_path, output_path):
                image = Image.open(input_path).convert("RGB")
                image.save(output_path)
                return image, {
                    "status": "partial",
                    "watermarks_found": 1,
                }

        monkeypatch.setattr(api, "_pipeline", PartialPipeline())
        with open(sample_image, "rb") as file:
            response = client.post(
                "/api/remove",
                files=[("files", ("partial.png", file, "image/png"))],
            )

        result = response.json()["results"][0]
        assert result["status"] == "error"
        assert result["download_url"] is None
        assert "Residual watermark" in result["error"]

    def test_remove_no_files(self, client):
        resp = client.post("/api/remove")
        assert resp.status_code == 422  # validation error

    def test_download_nonexistent(self, client):
        resp = client.get("/api/download/nonexistent.png")
        assert resp.status_code == 404

    def test_download_batch(self, client):
        from opennomark.api import OUTPUT_DIR

        first_id = uuid.uuid4().hex[:8]
        second_id = uuid.uuid4().hex[:8]
        first = OUTPUT_DIR / f"{first_id}_clean.png"
        second = OUTPUT_DIR / f"{second_id}_clean.png"
        Image.new("RGB", (24, 24), color=(20, 40, 60)).save(first)
        Image.new("RGB", (24, 24), color=(60, 40, 20)).save(second)

        try:
            resp = client.post(
                "/api/download-batch",
                json={
                    "items": [
                        {"job_id": first_id, "filename": "..\\same.png"},
                        {"job_id": second_id, "filename": "same.png"},
                    ]
                },
            )
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/zip"
            with zipfile.ZipFile(io.BytesIO(resp.content)) as bundle:
                assert bundle.namelist() == ["clean_same.png", "clean_same_2.png"]
        finally:
            first.unlink(missing_ok=True)
            second.unlink(missing_ok=True)

    def test_download_batch_rejects_unknown_jobs(self, client):
        resp = client.post(
            "/api/download-batch",
            json={"items": [{"job_id": "not-a-job", "filename": "image.png"}]},
        )
        assert resp.status_code == 404

    def test_remove_and_download(self, client, real_gemini_image):
        """E2E API: upload real image, get cleaned result, download it."""
        with open(real_gemini_image, "rb") as f:
            resp = client.post(
                "/api/remove",
                files=[("files", ("gemini.png", f, "image/png"))],
            )
        assert resp.status_code == 200
        data = resp.json()
        r = data["results"][0]
        assert r["status"] == "cleaned"
        assert r["download_url"] is not None

        # Download the cleaned image
        dl_resp = client.get(r["download_url"])
        assert dl_resp.status_code == 200
        # Verify it's a valid image
        img = Image.open(io.BytesIO(dl_resp.content))
        assert img.size[0] > 0
