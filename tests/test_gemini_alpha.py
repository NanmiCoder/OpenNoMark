"""Regression tests for the dedicated Gemini catalog detector and mask."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from opennomark.gemini_alpha import (
    _load_alpha,
    create_gemini_mask,
    detect_gemini_watermark,
)


def _background(width, height):
    yy, xx = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:, :, 0] = 35 + xx * 0.08 + 8 * np.sin(yy / 31)
    image[:, :, 1] = 55 + yy * 0.05 + 7 * np.cos(xx / 27)
    image[:, :, 2] = 70 + (xx + yy) * 0.025
    return np.clip(image, 0, 255).astype(np.uint8)


def _add_watermark(image, logo_size, margin):
    out = image.astype(np.float32).copy()
    alpha = _load_alpha(logo_size)[:, :, None]
    height, width = image.shape[:2]
    x = width - margin - logo_size
    y = height - margin - logo_size
    patch = out[y:y + logo_size, x:x + logo_size]
    out[y:y + logo_size, x:x + logo_size] = alpha * 255 + (1 - alpha) * patch
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), (x, y)


def test_detects_current_96px_192px_margin_layout():
    image, (x, y) = _add_watermark(_background(2048, 1024), 96, 192)
    detection = detect_gemini_watermark(image)

    assert detection["found"] is True
    assert detection["layout"] == "gemini_2k_20260520"
    assert detection["logo_size"] == 96
    assert abs(detection["x"] - x) <= 1
    assert abs(detection["y"] - y) <= 1


def test_detects_legacy_48px_32px_margin_layout():
    image, (x, y) = _add_watermark(_background(480, 640), 48, 32)
    detection = detect_gemini_watermark(image)

    assert detection["found"] is True
    assert detection["layout"] == "gemini_1k_current"
    assert detection["logo_size"] == 48
    assert abs(detection["x"] - x) <= 1
    assert abs(detection["y"] - y) <= 1


def test_rejects_catalog_positions_without_watermark():
    image = Image.fromarray(_background(800, 640))
    detection = detect_gemini_watermark(image)
    assert detection["found"] is False


def test_gemini_mask_is_tight_and_positioned():
    image, _ = _add_watermark(_background(2048, 1024), 96, 192)
    detection = detect_gemini_watermark(image)
    mask = np.asarray(create_gemini_mask(image.size, detection))

    assert mask.shape == (1024, 2048)
    assert mask.max() == 255
    assert np.count_nonzero(mask) < 96 * 96
    assert mask[:detection["y"] - 1].max() == 0
    assert mask[:, :detection["x"] - 1].max() == 0


def test_many_images_all_match_new_catalog_layout():
    image_dir = Path(__file__).resolve().parents[1] / "gemini_images" / "many_images"
    paths = sorted(image_dir.glob("*.png"))
    if not paths:
        pytest.skip("many_images dataset is not available")

    failures = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        detection = detect_gemini_watermark(image, source_hint=path.name)
        if not detection.get("found") or detection.get("layout") != "gemini_2k_20260520":
            failures.append((path.name, detection))
    assert failures == []
