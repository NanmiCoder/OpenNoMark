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


class FakeTextDetector:
    def __init__(self, accepted=None, *, overflow=False):
        self.accepted = list(accepted or [])
        self.detect_calls = 0
        self.last_filter_report = {
            "overflow": overflow,
            "reasons": ["max_regions"] if overflow else [],
        }

    def detect(self, image):
        self.detect_calls += 1
        return list(self.accepted)

    def filter_watermarks(self, proposals, width, height):
        return [] if self.last_filter_report["overflow"] else list(proposals)


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
        detector_factory=lambda: detector,
        text_detector_factory=lambda: FakeTextDetector(),
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
            "detection_tier": "corner_signature",
        }
    ]
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: _spatial_detection(confidence=0.21),
    )
    image = Image.new("RGB", (720, 1280), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: FakeTextDetector(),
    ).localize(image)

    assert evidence["arbitration"] == "spatial_different_corner"
    assert len(regions) == 1
    assert regions[0].source == "spatial_template"
    assert regions[0].box == [569.0, 1131.0, 665.0, 1227.0]


def test_spatial_template_keeps_nonoverlapping_generic_region(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: [
        {
            "box": [5.0, 1245.0, 130.0, 1273.0],
            "label": "brand watermark",
            "score": 0.8,
            "detection_tier": "generic_anywhere",
        }
    ]
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: _spatial_detection(confidence=0.21),
    )
    image = Image.new("RGB", (720, 1280), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: FakeTextDetector(),
    ).localize(image)

    assert evidence["arbitration"] == "spatial_different_corner"
    assert evidence["accepted_regions"] == 2
    assert [region.source for region in regions] == [
        "spatial_template",
        "open_vocabulary",
    ]
    assert regions[1].details["detector_profile"] == "generic_anywhere"


def test_overlapping_semantic_evidence_keeps_precise_mask(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: [
        {
            "box": [570.0, 1132.0, 664.0, 1226.0],
            "label": "logo watermark",
            "score": 0.7,
            "supporting_labels": ["logo watermark", "watermark"],
            "detection_tier": "generic_anywhere",
        }
    ]
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: _spatial_detection(),
    )
    image = Image.new("RGB", (720, 1280), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: FakeTextDetector(),
    ).localize(image)

    assert evidence["arbitration"] == "spatial_overlapping_evidence"
    assert evidence["accepted_regions"] == 1
    assert regions[0].source == "spatial_template"
    assert regions[0].method == "shape_mask"
    assert regions[0].details["validation_sources"] == [
        "spatial_template",
        "open_vocabulary",
    ]
    assert regions[0].details["supporting_labels"] == [
        "logo watermark",
        "watermark",
    ]


def test_ocr_polygon_replaces_overlapping_generic_box(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: [
        {
            "box": [250.0, 500.0, 550.0, 580.0],
            "label": "text watermark",
            "score": 0.4,
            "detection_tier": "generic_anywhere",
        }
    ]
    text_detector = FakeTextDetector(
        [
            {
                "box": [270.0, 510.0, 530.0, 570.0],
                "polygon": [
                    [270.0, 510.0],
                    [530.0, 510.0],
                    [530.0, 570.0],
                    [270.0, 570.0],
                ],
                "detection_score": 0.92,
                "text": "SAMPLE",
                "recognition_score": 0.95,
                "evidence": ["strong_lexical:sample"],
                "detection_tier": "generic_ocr",
            }
        ]
    )
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: {"found": False},
    )
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: text_detector,
    ).localize(image)

    assert evidence["experts"] == ["open_vocabulary", "ocr_text"]
    assert evidence["accepted_regions"] == 1
    assert regions[0].source == "ocr_text"
    assert regions[0].method == "polygon_mask"
    assert regions[0].details["text"] == "SAMPLE"
    assert regions[0].details["validation_sources"] == [
        "ocr_text",
        "open_vocabulary",
    ]
    assert regions[0].mask.getbbox() is not None


def test_unconfirmed_generic_semantic_region_is_not_removed(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: [
        {
            "box": [250.0, 500.0, 550.0, 580.0],
            "label": "text watermark",
            "score": 0.4,
            "detection_tier": "generic_anywhere",
        }
    ]
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: {"found": False},
    )
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: FakeTextDetector(),
    ).localize(image)

    assert regions == []
    assert evidence["suppressed_generic_regions"] == 1


def test_ocr_fallback_localizes_strong_text_when_semantic_detector_misses(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: []
    text_detector = FakeTextDetector(
        [
            {
                "box": [350.0, 400.0, 450.0, 620.0],
                "polygon": [
                    [350.0, 400.0],
                    [410.0, 400.0],
                    [450.0, 620.0],
                    [390.0, 620.0],
                ],
                "detection_score": 0.88,
                "text": "PROOF",
                "recognition_score": 0.91,
                "evidence": ["strong_lexical:proof"],
                "detection_tier": "generic_ocr",
            }
        ]
    )
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: {"found": False},
    )
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: text_detector,
    ).localize(image)

    assert text_detector.detect_calls == 1
    assert evidence["total_proposals"] == 1
    assert len(regions) == 1
    assert regions[0].source == "ocr_text"
    assert regions[0].box == [350.0, 400.0, 450.0, 620.0]


def test_ocr_overflow_is_reported_as_blocked_automatic_removal(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: []
    text_detector = FakeTextDetector([], overflow=True)
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: {"found": False},
    )
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: text_detector,
    ).localize(image)

    assert regions == []
    assert evidence["safety"] == {
        "automatic_removal_blocked": True,
        "overflow": [{"expert": "ocr_text", "reason": "max_regions"}],
    }


def test_corner_overflow_is_reported_as_blocked_automatic_removal(monkeypatch):
    detector = FakeDetector()
    detector.detect = lambda image: []
    detector.last_filter_report = {"corner_regions_truncated": True}
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: {"found": False},
    )
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))

    regions, evidence = WatermarkLocalizer(
        detector_factory=lambda: detector,
        text_detector_factory=lambda: FakeTextDetector(),
    ).localize(image)

    assert regions == []
    assert evidence["safety"] == {
        "automatic_removal_blocked": True,
        "overflow": [{"expert": "open_vocabulary", "reason": "max_regions"}],
    }


def test_residual_check_uses_recorded_supporting_experts(monkeypatch):
    detector = FakeDetector()
    monkeypatch.setattr(
        "opennomark.localizer.detect_gemini_watermark",
        lambda image: {"found": False},
    )
    image = Image.new("RGB", (800, 1200), color=(32, 48, 64))
    original = LocalizedWatermark(
        box=[650.0, 1130.0, 790.0, 1180.0],
        score=0.9,
        source="spatial_template",
        method="shape_mask",
        mask=Image.new("L", image.size, 0),
        details={
            "validation_sources": ["spatial_template", "open_vocabulary"],
        },
    )

    residuals = WatermarkLocalizer(
        detector_factory=lambda: detector
    ).localize_residuals(image, [original])

    assert detector.detect_calls == 1
    assert len(residuals) == 1
    assert residuals[0].source == "open_vocabulary"
