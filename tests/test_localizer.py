"""Unit tests for the unified localization contract."""

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
    assert regions[0].as_metadata() == {
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
