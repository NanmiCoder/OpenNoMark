"""Unit tests for watermark detector."""

import json

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
        # Calibrated corner signatures remain ahead of the stricter generic
        # candidate, even when the generic candidate has a higher raw score.
        assert [item["detection_tier"] for item in filtered] == [
            "corner_signature",
            "corner_signature",
            "generic_anywhere",
        ]
        assert [item["box"] for item in filtered] == [
            [650.0, 1150.0, 790.0, 1190.0],
            [10.0, 10.0, 160.0, 60.0],
            [300.0, 500.0, 500.0, 560.0],
        ]

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
        assert filtered[0]["detection_tier"] == "corner_signature"
        assert filtered[0]["supporting_labels"] == ["brand watermark"]

    def test_filter_rejects_large_boxes(self, detector):
        boxes = [
            {"box": [0, 0, 600, 400], "label": "watermark", "score": 0.5},
        ]
        filtered = detector.filter_watermarks(boxes, 800, 1200)
        assert len(filtered) == 0

    def test_filter_empty_input(self, detector):
        assert detector.filter_watermarks([], 800, 1200) == []

    def test_generic_accepts_high_confidence_center_watermark(self, detector):
        filtered = detector.filter_watermarks(
            [
                {
                    "box": [300, 500, 500, 560],
                    "label": "brand watermark",
                    "score": 0.314,
                }
            ],
            800,
            1200,
        )

        assert len(filtered) == 1
        assert filtered[0]["detection_tier"] == "generic_anywhere"
        assert filtered[0]["box"] == [300.0, 500.0, 500.0, 560.0]

    def test_generic_rejects_weak_center_watermark(self, detector):
        filtered = detector.filter_watermarks(
            [
                {
                    "box": [300, 500, 500, 560],
                    "label": "brand watermark",
                    "score": 0.199,
                }
            ],
            800,
            1200,
        )

        assert filtered == []

    def test_generic_accepts_overlapping_watermark_prompt_agreement(self, detector):
        filtered = detector.filter_watermarks(
            [
                {
                    "box": [300, 500, 500, 560],
                    "label": "watermark",
                    "score": 0.15,
                },
                {
                    "box": [304, 503, 498, 558],
                    "label": "text watermark",
                    "score": 0.06,
                },
            ],
            800,
            1200,
        )

        assert len(filtered) == 1
        assert filtered[0]["detection_tier"] == "generic_anywhere"
        assert filtered[0]["supporting_labels"] == ["text watermark", "watermark"]

    def test_generic_accepts_square_and_multiple_regions(self, detector):
        filtered = detector.filter_watermarks(
            [
                {
                    "box": [300, 300, 400, 400],
                    "label": "logo watermark",
                    "score": 0.25,
                },
                {
                    "box": [600, 500, 720, 620],
                    "label": "copyright watermark",
                    "score": 0.22,
                },
            ],
            1000,
            1000,
        )

        assert len(filtered) == 2
        assert all(item["detection_tier"] == "generic_anywhere" for item in filtered)
        assert [item["box"] for item in filtered] == [
            [300.0, 300.0, 400.0, 400.0],
            [600.0, 500.0, 720.0, 620.0],
        ]

    def test_plain_logo_cannot_trigger_generic_removal(self, detector):
        filtered = detector.filter_watermarks(
            [{"box": [300, 300, 420, 420], "label": "logo", "score": 0.9}],
            1000,
            1000,
        )

        assert filtered == []

    def test_plain_object_prompt_can_only_expand_supported_cluster(self, detector):
        filtered = detector.filter_watermarks(
            [
                {
                    "box": [400, 400, 500, 450],
                    "label": "watermark",
                    "score": 0.25,
                },
                {
                    "box": [390, 390, 520, 465],
                    "label": "symbol",
                    "score": 0.18,
                },
            ],
            1000,
            1000,
        )

        assert len(filtered) == 1
        assert filtered[0]["box"] == [390.0, 390.0, 520.0, 465.0]
        assert filtered[0]["label"] == "watermark"
        assert filtered[0]["box_source_label"] == "symbol"
        assert filtered[0]["supporting_labels"] == ["symbol", "watermark"]
        json.dumps(filtered)

    def test_transitive_cluster_cannot_replace_primary_with_remote_box(self, detector):
        filtered = detector.filter_watermarks(
            [
                {
                    "box": [850, 10, 950, 50],
                    "label": "brand watermark",
                    "score": 0.40,
                },
                {
                    "box": [300, 400, 500, 450],
                    "label": "brand watermark",
                    "score": 0.30,
                },
                {
                    # This invalid full-image proposal joins the two valid
                    # boxes into one transitive cluster, but must never allow
                    # the remote center box to replace the corner primary.
                    "box": [0, 0, 1000, 1000],
                    "label": "copyright watermark",
                    "score": 0.21,
                },
            ],
            1000,
            1000,
        )

        assert len(filtered) == 1
        assert filtered[0]["box"] == [850.0, 10.0, 950.0, 50.0]
        assert filtered[0]["detection_tier"] == "corner_signature"

    def test_prompt_preference_is_per_cluster_not_global(self, detector):
        filtered = detector.filter_watermarks(
            [
                {
                    "box": [650, 1150, 790, 1190],
                    "label": "watermark",
                    "score": 0.4,
                },
                {
                    "box": [300, 500, 500, 560],
                    "label": "brand watermark",
                    "score": 0.31,
                },
            ],
            800,
            1200,
        )

        assert len(filtered) == 2
        assert [item["label"] for item in filtered] == [
            "watermark",
            "brand watermark",
        ]

    def test_generic_area_overflow_rejects_entire_generic_lane(self, detector):
        boxes = [
            {
                "box": [820, 940, 980, 980],
                "label": "watermark",
                "score": 0.50,
            },
            *[
                {
                    "box": box,
                    "label": "text watermark",
                    "score": score,
                }
                for box, score in [
                    ([100, 300, 310, 510], 0.40),
                    ([390, 300, 600, 510], 0.39),
                    ([680, 300, 890, 510], 0.38),
                ]
            ],
        ]

        filtered = detector.filter_watermarks(boxes, 1000, 1000)

        assert len(filtered) == 1
        assert filtered[0]["detection_tier"] == "corner_signature"
        assert detector.last_filter_report["generic_overflow"] is True
        assert detector.last_filter_report["overflow_reason"] == "area_budget"
        assert detector.last_filter_report["accepted_corner_regions"] == 1
        assert detector.last_filter_report["accepted_generic_regions"] == 0
        json.dumps(detector.last_filter_report)

    def test_generic_region_count_overflow_rejects_entire_generic_lane(self, detector):
        boxes = [
            {
                "box": [100 + index * 150, 400, 180 + index * 150, 480],
                "label": "logo watermark",
                "score": 0.40 - index * 0.01,
            }
            for index in range(5)
        ]

        assert detector.filter_watermarks(boxes, 1000, 1000) == []
        assert detector.last_filter_report["generic_overflow"] is True
        assert detector.last_filter_report["overflow_reason"] == "max_regions"

    def test_corner_region_count_overflow_rejects_all_corner_regions(self, detector):
        boxes = [
            {
                "box": box,
                "label": "brand watermark",
                "score": 0.50,
            }
            for box in [
                [0, 960, 60, 990],
                [70, 960, 130, 990],
                [870, 960, 930, 990],
                [940, 960, 1000, 990],
                [0, 10, 60, 40],
            ]
        ]

        assert detector.filter_watermarks(boxes, 1000, 1000) == []
        assert detector.last_filter_report["corner_regions_truncated"] is True
        assert detector.last_filter_report["accepted_corner_regions"] == 0

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
