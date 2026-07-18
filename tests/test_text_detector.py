"""Unit tests for the independent PP-OCRv5 watermark proposal expert."""

import json

import pytest
import torch
from PIL import Image

from opennomark.text_detector import (
    DEFAULT_DETECTION_MODEL_ID,
    DEFAULT_RECOGNITION_MODEL_ID,
    TextWatermarkDetector,
)


class FakeBatch(dict):
    def to(self, device):
        self.device = torch.device(device)
        return self


class FakeDetectionProcessor:
    def __call__(self, *, images, return_tensors):
        assert isinstance(images, Image.Image)
        assert return_tensors == "pt"
        return FakeBatch(
            pixel_values=torch.zeros((1, 3, 16, 16)),
            target_sizes=torch.tensor([[images.height, images.width]]),
        )

    def post_process_object_detection(self, outputs, *, target_sizes):
        assert outputs == {"kind": "detection"}
        assert target_sizes.shape == (1, 2)
        return [
            {
                "boxes": torch.tensor(
                    [
                        [[10.0, 12.0], [110.0, 12.0], [110.0, 42.0], [10.0, 42.0]],
                        [[20.0, 50.0], [120.0, 50.0], [120.0, 80.0], [20.0, 80.0]],
                    ]
                ),
                "scores": torch.tensor([0.91, 0.23]),
            }
        ]


class FakeRecognitionProcessor:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, images, return_tensors):
        assert isinstance(images, Image.Image)
        assert return_tensors == "pt"
        self.calls += 1
        return FakeBatch(pixel_values=torch.zeros((1, 3, 8, 24)))

    def post_process_text_recognition(self, outputs):
        assert outputs == {"kind": "recognition"}
        return [{"text": "AI生成", "score": torch.tensor(0.94)}]


class FakeModel:
    def __init__(self, kind):
        self.kind = kind
        self.device = None
        self.evaluated = False

    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        self.evaluated = True
        return self

    def __call__(self, **inputs):
        assert "target_sizes" not in inputs
        assert "pixel_values" in inputs
        return {"kind": self.kind}


def proposal(text, box, detection_score=0.9, recognition_score=0.9):
    return {
        "box": box,
        "polygon": [
            [box[0], box[1]],
            [box[2], box[1]],
            [box[2], box[3]],
            [box[0], box[3]],
        ],
        "detection_score": detection_score,
        "text": text,
        "recognition_score": recognition_score,
    }


def test_detect_is_lazy_uses_real_api_shape_and_is_json_safe():
    created = []
    detection_processor = FakeDetectionProcessor()
    recognition_processor = FakeRecognitionProcessor()
    detection_model = FakeModel("detection")
    recognition_model = FakeModel("recognition")

    def factory(value):
        def create(model_id):
            created.append((value, model_id))
            return value

        return create

    detector = TextWatermarkDetector(
        device="cpu",
        detection_processor_factory=factory(detection_processor),
        detection_model_factory=factory(detection_model),
        recognition_processor_factory=factory(recognition_processor),
        recognition_model_factory=factory(recognition_model),
    )
    assert created == []
    assert detector.detection_model is None
    assert detector.recognition_model is None

    proposals = detector.detect(Image.new("RGB", (200, 100), "white"))

    assert len(proposals) == 1
    assert proposals[0] == {
        "box": [10.0, 12.0, 110.0, 42.0],
        "polygon": [
            [10.0, 12.0],
            [110.0, 12.0],
            [110.0, 42.0],
            [10.0, 42.0],
        ],
        "detection_score": pytest.approx(0.91),
        "text": "AI生成",
        "recognition_score": pytest.approx(0.94),
    }
    assert recognition_processor.calls == 1
    assert [value for value, _ in created] == [
        detection_processor,
        detection_model,
        recognition_processor,
        recognition_model,
    ]
    assert [model_id for _, model_id in created] == [
        DEFAULT_DETECTION_MODEL_ID,
        DEFAULT_DETECTION_MODEL_ID,
        DEFAULT_RECOGNITION_MODEL_ID,
        DEFAULT_RECOGNITION_MODEL_ID,
    ]
    assert detection_model.evaluated is True
    assert recognition_model.evaluated is True
    assert all(
        value.device == torch.device("cpu")
        for value in (detection_model, recognition_model)
    )
    json.dumps(proposals, ensure_ascii=False)


@pytest.mark.parametrize(
    "text",
    [
        "SAMPLE",
        "DO NOT COPY",
        "Confidential",
        "AI generated",
        "Generated with Example",
        "AI生成",
    ],
)
def test_filter_accepts_strong_lexical_terms_in_center(text):
    detector = TextWatermarkDetector(device="cpu")
    result = detector.filter_watermarks(
        [proposal(text, [400, 450, 600, 500])], 1000, 1000
    )

    assert len(result) == 1
    assert result[0]["source"] == "generic_ocr"
    assert result[0]["detection_tier"] == "generic_ocr"
    assert result[0]["evidence"][0].startswith("strong_lexical:")
    assert detector.last_filter_report["overflow"] is False
    json.dumps(result, ensure_ascii=False)
    json.dumps(detector.last_filter_report)


def test_filter_rejects_ordinary_scene_text():
    detector = TextWatermarkDetector(device="cpu")
    assert detector.filter_watermarks(
        [proposal("CENTRAL STATION ENTRANCE", [380, 420, 620, 480])],
        1000,
        1000,
    ) == []


@pytest.mark.parametrize(
    ("text", "evidence"),
    [
        ("https://example.com/gallery", "weak_lexical:url"),
        ("@example_creator", "weak_lexical:handle"),
        ("2026-07-18", "weak_lexical:date"),
    ],
)
def test_filter_accepts_weak_attribution_only_at_edge(text, evidence):
    detector = TextWatermarkDetector(device="cpu")
    result = detector.filter_watermarks(
        [proposal(text, [4, 500, 240, 540])], 1000, 1000
    )

    assert len(result) == 1
    assert "edge_geometry" in result[0]["evidence"]
    assert evidence in result[0]["evidence"]


def test_filter_rejects_url_in_center():
    detector = TextWatermarkDetector(device="cpu")
    assert detector.filter_watermarks(
        [proposal("www.example.com", [400, 450, 600, 490])], 1000, 1000
    ) == []


def test_filter_has_stable_priority_order_within_budget():
    detector = TextWatermarkDetector(device="cpu")
    proposals = [
        proposal("https://example.com", [2, 700, 202, 730], recognition_score=0.99),
        proposal("SAMPLE", [500, 500, 650, 540], recognition_score=0.75),
        proposal("DRAFT", [300, 300, 450, 340], recognition_score=0.90),
    ]

    first = detector.filter_watermarks(proposals, 1000, 1000)
    second = detector.filter_watermarks(proposals, 1000, 1000)

    assert [item["text"] for item in first] == [
        "DRAFT",
        "SAMPLE",
        "https://example.com",
    ]
    assert first == second
    assert detector.last_filter_report["total_area_ratio"] == pytest.approx(0.018)


def test_filter_rejects_entire_repeated_set_when_region_count_overflows():
    detector = TextWatermarkDetector(device="cpu", max_regions=4)
    tiled = [
        proposal("SAMPLE", [20 + index * 170, 100, 120 + index * 170, 130])
        for index in range(5)
    ]

    assert detector.filter_watermarks(tiled, 1000, 1000) == []
    assert detector.last_filter_report["overflow"] is True
    assert detector.last_filter_report["candidate_count"] == 5
    assert detector.last_filter_report["accepted_count"] == 0
    assert "max_regions" in detector.last_filter_report["reasons"]
    json.dumps(detector.last_filter_report)


def test_filter_rejects_entire_set_when_area_budget_overflows():
    detector = TextWatermarkDetector(device="cpu", max_total_area_ratio=0.12)
    candidates = [
        proposal("SAMPLE", [50, 100, 350, 400]),
        proposal("PROOF", [500, 500, 700, 700]),
    ]

    assert detector.filter_watermarks(candidates, 1000, 1000) == []
    assert detector.last_filter_report["overflow"] is True
    assert detector.last_filter_report["total_area_ratio"] == pytest.approx(0.13)
    assert "max_total_area_ratio" in detector.last_filter_report["reasons"]
