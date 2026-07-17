"""Tests for FastAPI backend."""

import os
import io
import uuid
import zipfile
import pytest
from PIL import Image


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
