"""Unit tests for watermark detector."""

import pytest
from PIL import Image


class TestWatermarkDetector:
    """Test OWLv2-based watermark detection."""

    @pytest.fixture(scope="class")
    def detector(self):
        from opennomark.detector import WatermarkDetector
        return WatermarkDetector(device="cpu")

    def test_init(self, detector):
        assert detector.model is not None
        assert detector.processor is not None

    def test_detect_returns_list(self, detector, sample_image):
        image = Image.open(sample_image).convert("RGB")
        boxes = detector.detect(image)
        assert isinstance(boxes, list)
        for box in boxes:
            assert "box" in box
            assert "label" in box
            assert "score" in box
            assert len(box["box"]) == 4

    def test_filter_keeps_corner_only(self, detector):
        boxes = [
            {"box": [750, 1150, 790, 1190], "label": "icon", "score": 0.2},  # bottom-right corner
            {"box": [300, 500, 500, 700], "label": "icon", "score": 0.3},    # center - should be filtered
            {"box": [10, 10, 50, 50], "label": "logo", "score": 0.15},       # top-left corner
        ]
        filtered = detector.filter_watermarks(boxes, 800, 1200)
        # Center box should be removed, corner boxes kept
        centers = [(((b["box"][0]+b["box"][2])/2), ((b["box"][1]+b["box"][3])/2)) for b in filtered]
        for cx, cy in centers:
            assert (cx < 800 * 0.15 or cx > 800 * 0.85) and (cy < 1200 * 0.15 or cy > 1200 * 0.85)

    def test_filter_rejects_large_boxes(self, detector):
        boxes = [
            {"box": [0, 0, 600, 400], "label": "logo", "score": 0.5},  # too large
        ]
        filtered = detector.filter_watermarks(boxes, 800, 1200)
        assert len(filtered) == 0

    def test_filter_empty_input(self, detector):
        assert detector.filter_watermarks([], 800, 1200) == []

    def test_detect_real_gemini(self, detector, real_gemini_image):
        """Test detection on a real Gemini image."""
        image = Image.open(real_gemini_image).convert("RGB")
        boxes = detector.detect(image)
        filtered = detector.filter_watermarks(boxes, image.width, image.height)
        assert len(filtered) >= 1, "Should detect at least 1 watermark in Gemini image"

    def test_detect_real_doubao(self, detector, real_doubao_image):
        """Test detection on a real Doubao image."""
        image = Image.open(real_doubao_image).convert("RGB")
        boxes = detector.detect(image)
        filtered = detector.filter_watermarks(boxes, image.width, image.height)
        assert len(filtered) >= 1, "Should detect at least 1 watermark in Doubao image"
