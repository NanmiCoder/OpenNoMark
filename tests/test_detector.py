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
            {"box": [650, 1150, 790, 1190], "label": "watermark", "score": 0.2},
            {"box": [300, 500, 500, 560], "label": "watermark", "score": 0.3},
            {"box": [10, 10, 160, 60], "label": "brand watermark", "score": 0.15},
        ]
        filtered = detector.filter_watermarks(boxes, 800, 1200)
        # The strongest trusted corner proposal wins; the center is rejected.
        assert len(filtered) == 1
        assert filtered[0]["box"] == [650.0, 1150.0, 790.0, 1190.0]

    def test_filter_rejects_untrusted_corner_objects(self, detector):
        boxes = [
            {"box": [650, 20, 790, 70], "label": "badge", "score": 0.9},
            {"box": [650, 1130, 790, 1180], "label": "icon", "score": 0.8},
        ]
        assert detector.filter_watermarks(boxes, 800, 1200) == []

    def test_dedup_uses_supported_full_text_box_not_low_score_oversize(self, detector):
        boxes = [
            {
                "box": [718, 1207, 935, 1250],
                "label": "brand watermark",
                "score": 0.38,
            },
            {
                "box": [703, 1206, 942, 1265],
                "label": "brand watermark",
                "score": 0.24,
            },
            {
                "box": [697, 1152, 901, 1278],
                "label": "brand watermark",
                "score": 0.03,
            },
        ]

        filtered = detector.filter_watermarks(boxes, 960, 1280)

        assert len(filtered) == 1
        assert filtered[0]["box"] == [703.0, 1206.0, 942.0, 1265.0]

    def test_filter_rejects_large_boxes(self, detector):
        boxes = [
            {"box": [0, 0, 600, 400], "label": "watermark", "score": 0.5},
        ]
        filtered = detector.filter_watermarks(boxes, 800, 1200)
        assert len(filtered) == 0

    def test_filter_empty_input(self, detector):
        assert detector.filter_watermarks([], 800, 1200) == []

    def test_detect_real_gemini(self, detector, real_gemini_image):
        """The unified localizer detects Gemini without a filename hint."""
        from opennomark.localizer import WatermarkLocalizer

        image = Image.open(real_gemini_image).convert("RGB")
        regions, evidence = WatermarkLocalizer(
            device="cpu", detector_factory=lambda: detector
        ).localize(image)
        assert len(regions) == 1
        assert regions[0].source == "spatial_template"
        assert evidence["accepted_regions"] == 1

    def test_detect_real_doubao(self, detector, real_doubao_image):
        """Test detection on a real Doubao image."""
        image = Image.open(real_doubao_image).convert("RGB")
        boxes = detector.detect(image)
        filtered = detector.filter_watermarks(boxes, image.width, image.height)
        assert len(filtered) >= 1, "Should detect at least 1 watermark in Doubao image"
