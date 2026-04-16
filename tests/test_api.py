"""Tests for FastAPI backend."""

import os
import io
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

    def test_remove_no_files(self, client):
        resp = client.post("/api/remove")
        assert resp.status_code == 422  # validation error

    def test_download_nonexistent(self, client):
        resp = client.get("/api/download/nonexistent.png")
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
