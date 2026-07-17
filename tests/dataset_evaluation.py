"""Deterministic acceptance harness for the real example-image datasets.

This module deliberately lives under ``tests`` rather than ``opennomark``:
provider directory names and acceptance policy belong to the repository's
evaluation dataset, not to production routing.  Run it directly with, for
example::

    uv run python -m tests.dataset_evaluation --mode inventory
    uv run python -m tests.dataset_evaluation --mode localize --providers gemini
    uv run python -m tests.dataset_evaluation --mode full --output report.json

``localize`` avoids loading LaMa and is intended for rapid detector iteration.
``full`` is the release gate: it verifies localization metadata, meaningful
localized pixel changes, change containment, and absence of a residual
detection over the original watermark region.  Merely writing an output file
is never considered success.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image


DATASET_PROVIDERS = (
    "baidu",
    "doubao",
    "gemini",
    "jimeng",
    "kling",
    "qwen",
    "yuanbao",
)
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SCHEMA_VERSION = 1
MIN_REGION_AREA_RATIO = 0.000001
MAX_REGION_AREA_RATIO = 0.25
CHANGE_THRESHOLD = 2
MIN_CHANGED_PIXELS = 8
MIN_CHANGE_CONTAINMENT = 0.90
RESIDUAL_OVERLAP_THRESHOLD = 0.25
CORNER_RATIO = 0.30


class EvaluationRuntime(Protocol):
    """Adapter used by the evaluator; tests inject a small deterministic fake."""

    def localize(self, image: Image.Image, path: Path) -> dict[str, Any]: ...

    def remove(self, image: Image.Image, path: Path) -> tuple[Image.Image, dict[str, Any]]: ...


class ProductionRuntime:
    """Lazy adapter around the production detector and removal pipeline."""

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self._localizer = None
        self._pipeline = None

    def _production_localizer(self):
        if self._localizer is None:
            from opennomark.localizer import WatermarkLocalizer

            self._localizer = WatermarkLocalizer(device=self.device)
        return self._localizer

    def localize(self, image: Image.Image, path: Path) -> dict[str, Any]:
        """Run the exact filename-independent localizer used in production."""
        regions, evidence = self._production_localizer().localize(image)
        return {
            "status": "localized" if regions else "no_watermark",
            "regions": [region.as_metadata() for region in regions],
            "localization": evidence,
        }

    def remove(self, image: Image.Image, path: Path) -> tuple[Image.Image, dict[str, Any]]:
        if self._pipeline is None:
            from opennomark.pipeline import WatermarkRemovalPipeline

            self._pipeline = WatermarkRemovalPipeline(device=self.device, verbose=False)
            # Reuse the already-loaded localizer during full evaluation. Without
            # this, localization and removal can hold two OWLv2 copies.
            if self._localizer is not None:
                self._pipeline.localizer = self._localizer
        # The pipeline owns image decoding in production. Passing the path here
        # keeps the harness honest about the real entry point.
        return self._pipeline.process(os.fspath(path))


def inventory_examples(dataset_root: Path) -> list[dict[str, Any]]:
    """Return the stable, decoded inventory of original dataset cases.

    Checked-in reference outputs named ``clean_*`` are excluded at discovery,
    regardless of whether they are tracked or untracked.
    """
    root = Path(dataset_root)
    cases: list[dict[str, Any]] = []
    for provider in DATASET_PROVIDERS:
        provider_dir = root / provider
        if not provider_dir.is_dir():
            continue
        paths = sorted(
            (
                path
                for path in provider_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_SUFFIXES
                and not path.name.lower().startswith("clean_")
            ),
            key=lambda path: path.name.casefold(),
        )
        for path in paths:
            item: dict[str, Any] = {
                "id": f"{provider}/{path.name}",
                "provider": provider,
                "path": path,
            }
            try:
                with Image.open(path) as opened:
                    opened.verify()
                with Image.open(path) as opened:
                    width, height = opened.size
                    mode = opened.mode
                item.update(
                    {
                        "width": int(width),
                        "height": int(height),
                        "image_mode": mode,
                        "inventory_errors": [],
                    }
                )
            except (OSError, ValueError) as exc:
                item.update(
                    {
                        "width": 0,
                        "height": 0,
                        "image_mode": None,
                        "inventory_errors": [f"image_decode_failed:{type(exc).__name__}"],
                    }
                )
            cases.append(item)
    return sorted(cases, key=lambda item: item["id"].casefold())


def regions_from_metadata(metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize the unified regions contract plus legacy pipeline metadata."""
    if not isinstance(metadata, dict):
        return []

    candidates: list[dict[str, Any]] = []
    raw_regions = metadata.get("regions")
    if isinstance(raw_regions, list):
        candidates.extend(item for item in raw_regions if isinstance(item, dict))

    if not candidates and isinstance(metadata.get("boxes"), list):
        candidates.extend(
            {
                **item,
                "source": item.get("source", "legacy_boxes"),
                "method": item.get("method", item.get("label", "open_vocabulary")),
            }
            for item in metadata["boxes"]
            if isinstance(item, dict)
        )

    gemini = metadata.get("gemini_detection")
    if not candidates and isinstance(gemini, dict):
        position = gemini.get("position")
        size = gemini.get("logo_size")
        if isinstance(position, (list, tuple)) and len(position) == 2 and size is not None:
            x, y = position
            candidates.append(
                {
                    "box": [x, y, float(x) + float(size), float(y) + float(size)],
                    "score": gemini.get("confidence", 0.0),
                    "source": "gemini_catalog",
                    "method": "template",
                }
            )

    normalized: list[dict[str, Any]] = []
    for item in candidates:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            normalized.append(
                {
                    "box": box,
                    "score": _json_number(item.get("score", 0.0)),
                    "source": str(item.get("source", "unknown")),
                    "method": str(item.get("method", item.get("label", "unknown"))),
                }
            )
            continue
        normalized_item = {
            "box": [_json_number(value) for value in box],
            "score": _json_number(item.get("score", 0.0)),
            "source": str(item.get("source", "unknown")),
            "method": str(item.get("method", item.get("label", "unknown"))),
        }
        mask_box = item.get("mask_box")
        if isinstance(mask_box, (list, tuple)) and len(mask_box) == 4:
            normalized_item["mask_box"] = [_json_number(value) for value in mask_box]
        normalized.append(normalized_item)
    return normalized


def validate_regions(
    regions: Sequence[dict[str, Any]], width: int, height: int
) -> list[str]:
    """Validate that claimed regions are finite, bounded, and watermark-sized."""
    errors: list[str] = []
    if not regions:
        return ["no_regions_localized"]
    image_area = max(1, width * height)
    for index, region in enumerate(regions):
        box = region.get("box")
        prefix = f"region_{index}"
        score = region.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            errors.append(f"{prefix}:invalid_score")
        elif not 0 <= float(score) <= 1:
            errors.append(f"{prefix}:score_out_of_range")
        if not str(region.get("source", "")).strip() or region.get("source") == "unknown":
            errors.append(f"{prefix}:source_missing")
        if not str(region.get("method", "")).strip() or region.get("method") == "unknown":
            errors.append(f"{prefix}:method_missing")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            errors.append(f"{prefix}:invalid_box_shape")
            continue
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in box):
            errors.append(f"{prefix}:non_finite_box")
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 <= x1 or y2 <= y1:
            errors.append(f"{prefix}:empty_box")
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            errors.append(f"{prefix}:out_of_bounds")
            continue
        area_ratio = ((x2 - x1) * (y2 - y1)) / image_area
        if area_ratio < MIN_REGION_AREA_RATIO:
            errors.append(f"{prefix}:implausibly_small")
        if area_ratio > MAX_REGION_AREA_RATIO:
            errors.append(f"{prefix}:implausibly_large")
    return errors


def evaluate_dataset(
    dataset_root: Path,
    *,
    mode: str,
    providers: Sequence[str] | None = None,
    limit: int | None = None,
    device: str | None = None,
    runtime: EvaluationRuntime | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate the selected cases and return a deterministic JSON-safe report."""
    if mode not in {"inventory", "localize", "full"}:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1")

    inventory = inventory_examples(Path(dataset_root))
    requested = tuple(providers or DATASET_PROVIDERS)
    unknown = sorted(set(requested) - set(DATASET_PROVIDERS))
    if unknown:
        raise ValueError(f"unknown providers: {', '.join(unknown)}")
    selected = [item for item in inventory if item["provider"] in requested]
    if limit is not None:
        selected = selected[:limit]

    if runtime is None and mode != "inventory":
        runtime = ProductionRuntime(device=device)
    if results_dir is not None:
        Path(results_dir).mkdir(parents=True, exist_ok=True)

    evaluated = [
        _evaluate_case(
            item,
            mode=mode,
            runtime=runtime,
            results_dir=Path(results_dir) if results_dir is not None else None,
        )
        for item in selected
    ]
    inventory_ids = {item["id"] for item in inventory}
    selected_ids = {item["id"] for item in selected}
    scope_complete = selected_ids == inventory_ids

    providers_summary: dict[str, dict[str, Any]] = {}
    for provider in DATASET_PROVIDERS:
        provider_cases = [case for case in evaluated if case["provider"] == provider]
        passed = sum(bool(case["passed"]) for case in provider_cases)
        providers_summary[provider] = {
            "cases": len(provider_cases),
            "passed": passed,
            "failed": len(provider_cases) - passed,
        }

    passed_cases = sum(bool(case["passed"]) for case in evaluated)
    selected_scope_passed = bool(evaluated) and passed_cases == len(evaluated)
    all_cases_handled = (
        mode in {"localize", "full"}
        and scope_complete
        and selected_scope_passed
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "criteria": _criteria_for_mode(mode),
        "coverage": {
            "inventory_cases": len(inventory),
            "selected_cases": len(evaluated),
            "providers": list(requested),
            "limit": limit,
            "scope_complete": scope_complete,
        },
        "summary": {
            "passed_cases": passed_cases,
            "failed_cases": len(evaluated) - passed_cases,
            "selected_scope_passed": selected_scope_passed,
            "all_cases_handled": all_cases_handled,
            "providers": providers_summary,
        },
        "cases": evaluated,
    }


def _evaluate_case(
    item: dict[str, Any],
    *,
    mode: str,
    runtime: EvaluationRuntime | None,
    results_dir: Path | None,
) -> dict[str, Any]:
    base = {
        "id": item["id"],
        "provider": item["provider"],
        "width": item["width"],
        "height": item["height"],
        "image_mode": item["image_mode"],
    }
    failures = list(item["inventory_errors"])
    if mode == "inventory" or failures:
        base.update({"passed": not failures, "failures": failures})
        return base

    assert runtime is not None
    path = item["path"]
    try:
        with Image.open(path) as opened:
            original = opened.convert("RGB")
        localization_metadata = runtime.localize(original.copy(), path)
        input_regions = regions_from_metadata(localization_metadata)
        failures.extend(validate_regions(input_regions, original.width, original.height))
        expected_corners = _expected_corners(item["provider"], path.name)
        target_regions, unexpected_regions = _regions_at_expected_corners(
            input_regions,
            expected_corners,
            original.width,
            original.height,
        )
        if input_regions and not target_regions:
            failures.append("expected_watermark_location_not_localized")
        if unexpected_regions:
            failures.append("unexpected_regions_outside_watermark_location")
        base["localization"] = {
            "status": localization_metadata.get("status", "unknown"),
            "region_count": len(input_regions),
            "regions": input_regions,
            "expected_corners": sorted(expected_corners),
            "target_region_count": len(target_regions),
            "unexpected_region_count": len(unexpected_regions),
        }

        if mode == "full" and not failures:
            result, removal_metadata = runtime.remove(original.copy(), path)
            removal_regions = regions_from_metadata(removal_metadata)
            failures.extend(validate_regions(removal_regions, original.width, original.height))
            removal_targets, removal_unexpected = _regions_at_expected_corners(
                removal_regions,
                expected_corners,
                original.width,
                original.height,
            )
            if removal_regions and not removal_targets:
                failures.append("removal_metadata_missing_expected_location")
            if removal_unexpected:
                failures.append("removal_metadata_has_unexpected_regions")
            failures.extend(_validate_removal_metadata(removal_metadata))
            removal_metrics, metric_failures = _measure_removal(
                original, result, target_regions
            )
            failures.extend(metric_failures)

            residual_metadata = runtime.localize(result.copy(), path)
            residual_regions = regions_from_metadata(residual_metadata)
            overlapping = _overlapping_residuals(target_regions, residual_regions)
            if overlapping:
                failures.append("watermark_region_still_detected")

            output_relative_path = None
            if results_dir is not None:
                output_path = results_dir / item["provider"] / path.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.save(output_path)
                output_relative_path = f"{item['provider']}/{path.name}"

            base["removal"] = {
                "status": removal_metadata.get("status", "unknown"),
                "methods": list(removal_metadata.get("methods", [])),
                "watermarks_found": removal_metadata.get("watermarks_found", 0),
                "regions": removal_regions,
                "metrics": removal_metrics,
                "residual_region_count": len(residual_regions),
                "overlapping_residual_regions": overlapping,
                "saved_result": output_relative_path,
            }
    except Exception as exc:  # Keep a complete dataset report on model failures.
        failures.append(f"evaluation_error:{type(exc).__name__}:{exc}")

    base.update({"passed": not failures, "failures": failures})
    return base


def _validate_removal_metadata(metadata: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if metadata.get("status") != "cleaned":
        failures.append("removal_status_not_cleaned")
    watermarks_found = metadata.get("watermarks_found")
    if not isinstance(watermarks_found, int) or watermarks_found < 1:
        failures.append("watermarks_found_not_positive")
    methods = metadata.get("methods")
    if not isinstance(methods, list) or not methods:
        failures.append("removal_method_missing")
    return failures


def _expected_corners(provider: str, filename: str) -> set[str]:
    """Return ground-truth location classes for this repository's dataset.

    This is intentionally test-data routing. It must not be copied into the
    production detector. The three original Doubao reference cases use the
    older top-left badge; the newly supplied Doubao set and the other two
    providers use bottom-right marks.
    """
    if provider == "doubao" and filename.casefold().startswith("doubao_sample_"):
        return {"top_left"}
    return {"bottom_right"}


def _regions_at_expected_corners(
    regions: Sequence[dict[str, Any]],
    expected_corners: set[str],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    for region in regions:
        box = region.get("box")
        if not _valid_numeric_box(box):
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        center_x = (x1 + x2) / 2 / max(1, width)
        center_y = (y1 + y2) / 2 / max(1, height)
        horizontal = "left" if center_x <= CORNER_RATIO else (
            "right" if center_x >= 1 - CORNER_RATIO else "center"
        )
        vertical = "top" if center_y <= CORNER_RATIO else (
            "bottom" if center_y >= 1 - CORNER_RATIO else "center"
        )
        corner = f"{vertical}_{horizontal}"
        (target if corner in expected_corners else unexpected).append(region)
    return target, unexpected


def _measure_removal(
    original: Image.Image,
    result: Image.Image,
    input_regions: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    if result.size != original.size:
        return (
            {
                "same_dimensions": False,
                "changed_pixels": 0,
                "changed_pixel_ratio": 0.0,
                "change_containment": 0.0,
            },
            ["output_dimensions_changed"],
        )

    original_np = np.asarray(original.convert("RGB"), dtype=np.int16)
    result_np = np.asarray(result.convert("RGB"), dtype=np.int16)
    changed = np.max(np.abs(original_np - result_np), axis=2) > CHANGE_THRESHOLD
    changed_pixels = int(np.count_nonzero(changed))
    required_changes = max(MIN_CHANGED_PIXELS, int(original.width * original.height * 0.000001))
    if changed_pixels < required_changes:
        failures.append("output_has_no_meaningful_pixel_change")

    allowed = np.zeros((original.height, original.width), dtype=bool)
    for region in input_regions:
        mask_box = region.get("mask_box")
        box = mask_box if _valid_numeric_box(mask_box) else region["box"]
        # New metadata exposes the exact feathered-mask bounds. Legacy
        # callers only report detector boxes, so retain the historical safety
        # margin for those records.
        expansion = 2 if _valid_numeric_box(mask_box) else max(
            8, int(round(min(original.size) * 0.01))
        )
        x1 = max(0, int(math.floor(float(box[0]))) - expansion)
        y1 = max(0, int(math.floor(float(box[1]))) - expansion)
        x2 = min(original.width, int(math.ceil(float(box[2]))) + expansion)
        y2 = min(original.height, int(math.ceil(float(box[3]))) + expansion)
        allowed[y1:y2, x1:x2] = True
    contained_changes = int(np.count_nonzero(changed & allowed))
    containment = contained_changes / changed_pixels if changed_pixels else 0.0
    if changed_pixels and containment < MIN_CHANGE_CONTAINMENT:
        failures.append("pixel_changes_escape_localized_regions")

    metrics = {
        "same_dimensions": True,
        "changed_pixels": changed_pixels,
        "changed_pixel_ratio": _round_float(changed_pixels / changed.size),
        "change_containment": _round_float(containment),
        "required_changed_pixels": required_changes,
        "minimum_change_containment": MIN_CHANGE_CONTAINMENT,
    }
    return metrics, failures


def _overlapping_residuals(
    input_regions: Sequence[dict[str, Any]],
    residual_regions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    overlapping: list[dict[str, Any]] = []
    for residual in residual_regions:
        residual_box = residual.get("box")
        if not _valid_numeric_box(residual_box):
            continue
        maximum = max(
            (_intersection_over_reference(original["box"], residual_box) for original in input_regions),
            default=0.0,
        )
        if maximum >= RESIDUAL_OVERLAP_THRESHOLD:
            overlapping.append(
                {
                    "box": list(residual_box),
                    "overlap_with_original": _round_float(maximum),
                    "source": residual.get("source", "unknown"),
                    "method": residual.get("method", "unknown"),
                }
            )
    return overlapping


def _intersection_over_reference(reference: Sequence[float], other: Sequence[float]) -> float:
    rx1, ry1, rx2, ry2 = (float(value) for value in reference)
    ox1, oy1, ox2, oy2 = (float(value) for value in other)
    intersection_w = max(0.0, min(rx2, ox2) - max(rx1, ox1))
    intersection_h = max(0.0, min(ry2, oy2) - max(ry1, oy1))
    reference_area = max(0.0, (rx2 - rx1) * (ry2 - ry1))
    return intersection_w * intersection_h / reference_area if reference_area else 0.0


def _valid_numeric_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple))
        and len(box) == 4
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in box)
    )


def _criteria_for_mode(mode: str) -> dict[str, Any]:
    common = {
        "original_cases_only": "supported images excluding clean_*",
        "decoded_image_required": True,
    }
    if mode == "inventory":
        return common
    common.update(
        {
            "localized_region_required": True,
            "region_area_ratio": [MIN_REGION_AREA_RATIO, MAX_REGION_AREA_RATIO],
            "region_must_be_in_bounds": True,
            "region_score_source_method_required": True,
            "expected_dataset_location_required": True,
            "unexpected_location_regions_allowed": False,
            "corner_ratio": CORNER_RATIO,
        }
    )
    if mode == "full":
        common.update(
            {
                "removal_status": "cleaned",
                "positive_watermark_count": True,
                "removal_method_required": True,
                "minimum_changed_pixels": MIN_CHANGED_PIXELS,
                "minimum_change_containment": MIN_CHANGE_CONTAINMENT,
                "maximum_residual_overlap": RESIDUAL_OVERLAP_THRESHOLD,
                "output_dimensions_preserved": True,
            }
        )
    return common


def _json_number(value: Any) -> int | float | str | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _round_float(float(value)) if math.isfinite(float(value)) else str(value)
    return value


def _round_float(value: float) -> float:
    return round(float(value), 8)


def write_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output is None:
        print(payload, end="")
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(payload, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1] / "examples"
    parser.add_argument("--dataset-root", type=Path, default=default_root)
    parser.add_argument("--mode", choices=("inventory", "localize", "full"), default="localize")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=DATASET_PROVIDERS,
        default=list(DATASET_PROVIDERS),
        help="Dataset directories to evaluate; partial scopes never set all_cases_handled=true.",
    )
    parser.add_argument("--limit", type=int, help="Deterministic development sample after sorting.")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--output", type=Path, help="Write the JSON report instead of stdout.")
    parser.add_argument("--results-dir", type=Path, help="In full mode, optionally save cleaned images.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_dataset(
        args.dataset_root,
        mode=args.mode,
        providers=args.providers,
        limit=args.limit,
        device=args.device,
        results_dir=args.results_dir,
    )
    write_report(report, args.output)
    return 0 if report["summary"]["selected_scope_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
