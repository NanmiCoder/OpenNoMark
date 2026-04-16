"""Shared fixtures for tests."""

import os
import tempfile
import pytest
from PIL import Image


@pytest.fixture
def sample_image(tmp_path):
    """Create a simple test image with a fake watermark-like element in the corner."""
    img = Image.new("RGB", (800, 1200), color=(40, 40, 45))
    # Draw a small white rectangle in bottom-right corner to simulate watermark
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([750, 1150, 790, 1190], fill=(200, 200, 200))
    path = str(tmp_path / "test_image.png")
    img.save(path)
    return path


@pytest.fixture
def sample_images_dir(tmp_path):
    """Create a directory with multiple test images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    from PIL import ImageDraw
    for i, ext in enumerate(["png", "jpg", "jpeg"]):
        img = Image.new("RGB", (400, 600), color=(30 + i * 20, 30, 40))
        draw = ImageDraw.Draw(img)
        draw.rectangle([350, 550, 390, 590], fill=(180, 180, 180))
        img.save(str(img_dir / f"test_{i}.{ext}"))
    return str(img_dir)


@pytest.fixture
def output_dir(tmp_path):
    """Provide a clean output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return str(out)


@pytest.fixture
def real_gemini_image():
    """Return path to a real Gemini test image if available."""
    path = os.path.join(os.path.dirname(__file__), "..", "gemini_images", "Gemini_Generated_Image_ (4).png")
    if os.path.exists(path):
        return os.path.abspath(path)
    pytest.skip("Real Gemini test image not available")


@pytest.fixture
def real_doubao_image():
    """Return path to a real Doubao test image if available."""
    base = os.path.join(os.path.dirname(__file__), "..", "豆包")
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f.endswith(".jpeg"):
                return os.path.abspath(os.path.join(base, f))
    pytest.skip("Real Doubao test image not available")
