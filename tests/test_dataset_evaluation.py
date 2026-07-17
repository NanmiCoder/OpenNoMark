"""Tests for the real-example dataset acceptance harness."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from tests.dataset_evaluation import (
    evaluate_dataset,
    inventory_examples,
    main,
    regions_from_metadata,
    validate_regions,
)


def _make_dataset(root: Path, *, clean_reference: bool = True) -> Path:
    for index, provider in enumerate(("doubao", "gemini", "qwen")):
        directory = root / provider
        directory.mkdir(parents=True)
        Image.new("RGB", (100, 80), (20 + index, 30, 40)).save(directory / f"case_{index}.png")
        if clean_reference:
            Image.new("RGB", (100, 80), (20, 30, 40)).save(directory / f"clean_case_{index}.png")
    return root


class GoodRuntime:
    region = {"box": [70, 50, 90, 70], "score": 0.9, "source": "test", "method": "fake"}

    def localize(self, image, path):
        # A successful removal paints the first pixel of the claimed region red.
        cleaned = np.asarray(image)[55, 75, 0] == 220
        return {"status": "no_watermark" if cleaned else "localized", "regions": [] if cleaned else [self.region]}

    def remove(self, image, path):
        array = np.array(image)
        array[55:60, 75:80] = (220, 20, 20)
        return Image.fromarray(array), {
            "status": "cleaned",
            "watermarks_found": 1,
            "methods": ["fake_inpaint"],
            "regions": [self.region],
        }


class UnchangedRuntime(GoodRuntime):
    def remove(self, image, path):
        return image.copy(), {
            "status": "cleaned",
            "watermarks_found": 1,
            "methods": ["fake_inpaint"],
            "regions": [self.region],
        }


class EscapingRuntime(GoodRuntime):
    def remove(self, image, path):
        array = np.array(image)
        array[0:20, 0:20] = (220, 20, 20)
        return Image.fromarray(array), {
            "status": "cleaned",
            "watermarks_found": 1,
            "methods": ["fake_inpaint"],
            "regions": [self.region],
        }


class ResidualRuntime(GoodRuntime):
    def localize(self, image, path):
        return {"status": "localized", "regions": [self.region]}


class WrongCornerRuntime(GoodRuntime):
    region = {"box": [5, 5, 25, 25], "score": 0.9, "source": "test", "method": "fake"}


def test_inventory_is_stable_and_excludes_clean_references(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    inventory = inventory_examples(root)

    assert [item["id"] for item in inventory] == [
        "doubao/case_0.png",
        "gemini/case_1.png",
        "qwen/case_2.png",
    ]
    assert all(item["inventory_errors"] == [] for item in inventory)
    assert all(item["width"] == 100 and item["height"] == 80 for item in inventory)


def test_regions_contract_prefers_unified_regions_and_supports_legacy_metadata():
    unified = regions_from_metadata(
        {
            "regions": [{"box": [1, 2, 11, 12], "score": 0.876543219, "source": "new", "method": "mask"}],
            "boxes": [{"box": [20, 20, 30, 30]}],
        }
    )
    legacy_boxes = regions_from_metadata(
        {"boxes": [{"box": [3, 4, 13, 14], "score": 0.5, "label": "logo"}]}
    )
    legacy_gemini = regions_from_metadata(
        {"gemini_detection": {"position": [5, 6], "logo_size": 8, "confidence": 0.7}}
    )

    assert unified == [
        {"box": [1, 2, 11, 12], "score": 0.87654322, "source": "new", "method": "mask"}
    ]
    assert legacy_boxes[0]["method"] == "logo"
    assert legacy_boxes[0]["source"] == "legacy_boxes"
    assert legacy_gemini[0]["box"] == [5, 6, 13.0, 14.0]


def test_region_validation_rejects_empty_out_of_bounds_and_implausibly_large():
    assert validate_regions([], 100, 80) == ["no_regions_localized"]
    errors = validate_regions(
        [
            {"box": [10, 10, 10, 20], "score": 0.5, "source": "test", "method": "fake"},
            {"box": [-1, 0, 5, 5], "score": 0.5, "source": "test", "method": "fake"},
            {"box": [0, 0, 90, 70], "score": 0.5, "source": "test", "method": "fake"},
        ],
        100,
        80,
    )
    assert errors == [
        "region_0:empty_box",
        "region_1:out_of_bounds",
        "region_2:implausibly_large",
    ]


def test_region_validation_requires_complete_machine_readable_metadata():
    errors = validate_regions(
        [{"box": [10, 10, 20, 20], "score": 3, "source": "", "method": "unknown"}],
        100,
        80,
    )
    assert errors == [
        "region_0:score_out_of_range",
        "region_0:source_missing",
        "region_0:method_missing",
    ]


def test_localize_partial_scope_cannot_claim_all_cases_handled(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    report = evaluate_dataset(
        root,
        mode="localize",
        providers=["qwen"],
        runtime=GoodRuntime(),
    )

    assert report["coverage"] == {
        "inventory_cases": 3,
        "selected_cases": 1,
        "providers": ["qwen"],
        "limit": None,
        "scope_complete": False,
    }
    assert report["summary"]["selected_scope_passed"] is True
    assert report["summary"]["all_cases_handled"] is False


def test_localize_rejects_a_valid_box_in_the_wrong_dataset_corner(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    report = evaluate_dataset(
        root,
        mode="localize",
        providers=["qwen"],
        runtime=WrongCornerRuntime(),
    )

    assert report["cases"][0]["failures"] == [
        "expected_watermark_location_not_localized",
        "unexpected_regions_outside_watermark_location",
    ]


def test_full_gate_proves_changes_are_local_and_residual_is_gone(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    report = evaluate_dataset(root, mode="full", runtime=GoodRuntime())

    assert report["summary"]["all_cases_handled"] is True
    assert report["summary"]["passed_cases"] == 3
    for case in report["cases"]:
        assert case["removal"]["metrics"]["changed_pixels"] == 25
        assert case["removal"]["metrics"]["change_containment"] == 1.0
        assert case["removal"]["residual_region_count"] == 0


def test_full_gate_fails_an_unchanged_output_despite_success_metadata(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    report = evaluate_dataset(root, mode="full", runtime=UnchangedRuntime())

    assert report["summary"]["all_cases_handled"] is False
    assert all(
        "output_has_no_meaningful_pixel_change" in case["failures"]
        for case in report["cases"]
    )


def test_full_gate_fails_changes_outside_localized_region(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    report = evaluate_dataset(root, mode="full", runtime=EscapingRuntime())

    assert all(
        "pixel_changes_escape_localized_regions" in case["failures"]
        for case in report["cases"]
    )


def test_full_gate_fails_residual_detection_over_original_region(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    report = evaluate_dataset(root, mode="full", runtime=ResidualRuntime())

    assert all(
        "watermark_region_still_detected" in case["failures"]
        for case in report["cases"]
    )


def test_json_report_is_machine_readable_and_deterministic(tmp_path):
    root = _make_dataset(tmp_path / "examples")
    first = evaluate_dataset(root, mode="inventory")
    second = evaluate_dataset(root, mode="inventory")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)

    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--dataset-root",
            str(root),
            "--mode",
            "inventory",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1


def test_repository_inventory_covers_every_provider_and_excludes_clean_files():
    root = Path(__file__).resolve().parents[1] / "examples"
    inventory = inventory_examples(root)

    assert inventory
    assert {item["provider"] for item in inventory} == {"doubao", "gemini", "qwen"}
    counts = Counter(item["provider"] for item in inventory)
    assert counts["doubao"] >= 16
    assert counts["gemini"] >= 15
    assert counts["qwen"] >= 21
    assert all(not Path(item["id"]).name.lower().startswith("clean_") for item in inventory)
    assert all(item["inventory_errors"] == [] for item in inventory)
