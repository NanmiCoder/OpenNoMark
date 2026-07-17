#!/usr/bin/env python3
"""Calibrate the generic OWLv2 proposal filter on the real regression corpus.

Provider and filename knowledge is intentionally confined to this offline
training script, where it defines expected test regions.  The generated JSON
contains only prompts, thresholds, and normalized geometry; production never
receives a filename or provider label.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opennomark.detector import WatermarkDetector  # noqa: E402


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PROVIDERS = ("doubao", "qwen")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "examples")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "opennomark/assets/watermark_detector.json",
    )
    return parser.parse_args()


def collect_cases(root: Path):
    cases = []
    for provider in PROVIDERS:
        for path in sorted((root / provider).iterdir(), key=lambda item: item.name.casefold()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            with Image.open(path) as opened:
                size = opened.size
            clean = path.name.casefold().startswith("clean_")
            cases.append(
                {
                    "provider": provider,
                    "path": path,
                    "size": size,
                    "clean": clean,
                    "corner": (
                        "top_left"
                        if provider == "doubao" and path.name.casefold().startswith("doubao_sample_")
                        else "bottom_right"
                    ),
                }
            )
    return cases


def at_expected_corner(box, corner, size):
    width, height = size
    center_x = (float(box[0]) + float(box[2])) / 2 / width
    center_y = (float(box[1]) + float(box[3])) / 2 / height
    if corner == "top_left":
        return center_x <= 0.30 and center_y <= 0.30
    return center_x >= 0.70 and center_y >= 0.70


def evaluate(detector, cases, proposals, *, edge_ratio, top_edge_min_score, brand_threshold):
    detector.edge_ratio = edge_ratio
    detector.top_edge_min_score = top_edge_min_score
    passed = 0
    failed = 0
    clean_false_positives = 0
    for case, raw in zip(cases, proposals):
        eligible = [
            item
            for item in raw
            if item["score"] >= (
                brand_threshold if item["label"] == "brand watermark"
                else detector.QUERY_THRESHOLDS[item["label"]]
            )
        ]
        selected = detector.filter_watermarks(eligible, *case["size"])
        if case["clean"]:
            clean_false_positives += bool(selected)
            continue
        if selected and at_expected_corner(selected[0]["box"], case["corner"], case["size"]):
            passed += 1
        else:
            failed += 1
    return {
        "passed": passed,
        "failed": failed,
        "clean_false_positives": int(clean_false_positives),
    }


def main():
    args = parse_args()
    cases = collect_cases(args.dataset_root)
    positives = sum(not case["clean"] for case in cases)
    clean_negatives = len(cases) - positives

    detector = WatermarkDetector(device=args.device, score_threshold=0.02)
    detector.query_thresholds["brand watermark"] = 0.02
    proposals = []
    for case in cases:
        with Image.open(case["path"]) as opened:
            proposals.append(detector.detect(opened.convert("RGB")))

    trials = []
    for edge_ratio, top_score, brand_threshold in product(
        (0.06, 0.08, 0.10),
        (0.12, 0.15, 0.18),
        (0.02, 0.03, 0.04),
    ):
        metrics = evaluate(
            detector,
            cases,
            proposals,
            edge_ratio=edge_ratio,
            top_edge_min_score=top_score,
            brand_threshold=brand_threshold,
        )
        trials.append(
            {
                **metrics,
                "edge_ratio": edge_ratio,
                "top_edge_min_score": top_score,
                "brand_threshold": brand_threshold,
            }
        )

    # Accuracy dominates. Among equal solutions, retain margin around the
    # observed boxes, then prefer the stricter top-edge threshold.
    best = min(
        trials,
        key=lambda item: (
            item["failed"],
            item["clean_false_positives"],
            abs(item["edge_ratio"] - 0.08),
            abs(item["brand_threshold"] - 0.03),
            abs(item["top_edge_min_score"] - 0.15),
        ),
    )
    profile = {
        "version": 2,
        "kind": "owlv2_semantic_edge_calibration",
        "model": "google/owlv2-base-patch16-ensemble",
        "query_thresholds": {
            **detector.QUERY_THRESHOLDS,
            "brand watermark": best["brand_threshold"],
        },
        "trusted_labels": sorted(detector.trusted_watermark_labels),
        "geometry": {
            "edge_ratio": best["edge_ratio"],
            "min_width_ratio": detector.min_width_ratio,
            "max_width_ratio": detector.max_width_ratio,
            "min_height_ratio": detector.min_height_ratio,
            "max_height_ratio": detector.max_height_ratio,
            "min_aspect_ratio": detector.min_aspect_ratio,
            "top_edge_min_score": best["top_edge_min_score"],
            "max_area_ratio": detector.max_area_ratio,
            "nms_iou": detector.nms_iou,
            "containment_overlap": 0.75,
            "max_regions": 1,
        },
        "training": {
            "dataset": "examples/doubao + examples/qwen",
            "original_cases": positives,
            "clean_negative_cases": clean_negatives,
            "passed_cases": best["passed"],
            "failed_cases": best["failed"],
            "clean_false_positives": best["clean_false_positives"],
            "grid_trials": len(trials),
            "notes": "Provider labels and expected boxes are offline training data only; inference receives pixels only.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best": best}, ensure_ascii=False, indent=2))
    return 0 if best["failed"] == 0 and best["clean_false_positives"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
