"""Unit tests for the unified localization contract."""

import numpy as np
from PIL import Image

from opennomark.localizer import LocalizedWatermark, WatermarkLocalizer


class FakeDetector:
    def __init__(self):
        self.detect_calls = 0

    def detect(self, image):
        self.detect_calls += 1
        return [
            {
                "box": [650.0, 1130.0, 790.0, 1180.0],
                "label": "brand watermark",
                "score": 0.4,
            }
        ]

    def filter_watermarks(self, boxes, image_width, image_height):
        return boxes


def _spatial_detection(*, confidence=0.275):
    return {
        "found": True,
        "x": 569,
        "y": 1131,
        "logo_size": 96,
        "layout": "gemini_large_legacy",
        "spatial_score": 0.322,
        "gradient_score": 0.227,
        "confidence": confidence,
        "decision": "trained_match",
        "model_version": 1,
        "alpha_map": np.ones((96, 96), dtype=np.float32),
    }


def test_generic_region_has_serializable_metadata_and_local_mask():
    detector = FakeDetector()
    localizer = WatermarkLocalizer(detector_factory=lambda: detector)
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))

    regions, evidence = localizer.localize(image)

    assert detector.detect_calls == 1
    assert evidence == {
        "total_proposals": 1,
        "accepted_regions": 1,
        "experts": ["open_vocabulary"],
    }
    assert len(regions) == 1
    assert regions[0].mask.getbbox() is not None
    metadata = regions[0].as_metadata()
    assert metadata.pop("mask_box") == [
        float(value) for value in regions[0].mask.getbbox()
    ]
    assert metadata == {
        "box": [650.0, 1130.0, 790.0, 1180.0],
        "score": 0.4,
        "source": "open_vocabulary",
        "method": "box_mask",
        "details": {
            "label": "brand watermark",
            "raw_score": 0.4,
            "mask_padding": 6,
        },
    }


def test_residual_check_does_not_load_unrelated_expert():
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))
    original = LocalizedWatermark(
        box=[700.0, 1100.0, 748.0, 1148.0],
        score=0.9,
        source="spatial_template",
        method="shape_mask",
        mask=Image.new("L", image.size, 0),
    )
    localizer = WatermarkLocalizer(
        detector_factory=lambda: (_ for _ in ()).throw(AssertionError("OWLv2 loaded"))
    )

    assert localizer.localize_residuals(image, [original]) == []


def test_stronger_edge_text_overrides_nonoverlapping_spatial_false_positive(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: [
        {
            "box": [588.0, 1245.0, 713.0, 1273.0],
            "label": "brand watermark",
            "score": 0.28,
        }
    ]
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: _spatial_detection(),
    )
    image = Image.new("RGB", (720, 1280), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector
    ).localize(image)

    assert evidence["experts"] == ["spatial_template", "open_vocabulary"]
    assert evidence["arbitration"] == "semantic_override"
    assert regions[0].source == "open_vocabulary"
    assert regions[0].box == [588.0, 1245.0, 713.0, 1273.0]
    assert regions[0].details["suppressed_spatial_score"] == 0.275


def test_spatial_template_wins_when_semantic_proposal_is_in_another_corner(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: [
        {
            "box": [5.0, 1245.0, 130.0, 1273.0],
            "label": "brand watermark",
            "score": 0.8,
        }
    ]
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: _spatial_detection(confidence=0.21),
    )
    image = Image.new("RGB", (720, 1280), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector
    ).localize(image)

    assert evidence["arbitration"] == "spatial_different_corner"
    assert regions[0].source == "spatial_template"
    assert regions[0].box == [569.0, 1131.0, 665.0, 1227.0]
