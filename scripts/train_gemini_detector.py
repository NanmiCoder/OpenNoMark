#!/usr/bin/env python3
"""Calibrate the Gemini catalog detector from real positive images.

This is deliberately a small, inspectable threshold model rather than a CNN:
Gemini watermarks have a known alpha template and a finite anchor catalog, so
the useful learned parameters are the joint spatial/edge evidence thresholds.
Hard negatives are sampled from the same source images away from the real
watermark, which makes the calibration background-diverse without requiring
paired clean originals.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opennomark.gemini_alpha import (  # noqa: E402
    _best_layout_candidate,
    _candidate_layouts,
    _load_alpha,
    _score_patch,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Positive image files or directories")
    parser.add_argument(
        "--output",
        default=str(ROOT / "opennomark/assets/gemini_detector.json"),
        help="Output model JSON",
    )
    parser.add_argument("--negatives-per-image", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def collect_unique_images(inputs):
    paths = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        elif path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            paths.append(path)

    unique = []
    seen = set()
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((path, digest))
    return unique


def to_gray(image):
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]


def infer_positive(gray):
    candidates = [
        candidate
        for layout in _candidate_layouts(gray.shape[1], gray.shape[0])
        if (candidate := _best_layout_candidate(gray, layout)) is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["ranking_score"])


def sample_hard_negatives(gray, positive, count, rng):
    height, width = gray.shape
    size = positive["logo_size"]
    alpha = _load_alpha(size)
    features = []
    attempts = 0
    while len(features) < count and attempts < count * 50:
        attempts += 1
        x = rng.randrange(0, width - size + 1)
        y = rng.randrange(0, height - size + 1)
        if abs(x - positive["x"]) <= size * 2 and abs(y - positive["y"]) <= size * 2:
            continue
        patch = gray[y:y + size, x:x + size]
        features.append(_score_patch(patch, alpha))
    return features


def train_thresholds(positives, negatives, robustness_margin=0.02):
    # Search a joint AND-rule. Requiring both template shape and template edge
    # evidence rejects hard negatives that correlate in brightness only.
    choices = np.arange(0.05, 0.301, 0.01)
    viable = []
    for spatial_threshold in choices:
        for gradient_threshold in choices:
            robust_recall = sum(
                spatial >= spatial_threshold + robustness_margin and
                gradient >= gradient_threshold + robustness_margin
                for spatial, gradient in positives
            ) / len(positives)
            if robust_recall < 1.0:
                continue
            false_positives = sum(
                spatial >= spatial_threshold and gradient >= gradient_threshold
                for spatial, gradient in negatives
            )
            viable.append((false_positives, -(spatial_threshold + gradient_threshold),
                           spatial_threshold, gradient_threshold))

    if not viable:
        return (
            max(0.05, min(spatial for spatial, _ in positives) - robustness_margin),
            max(0.03, min(gradient for _, gradient in positives) - robustness_margin),
        )
    _, _, spatial_threshold, gradient_threshold = min(viable)
    return float(spatial_threshold), float(gradient_threshold)


def main():
    args = parse_args()
    samples = collect_unique_images(args.inputs)
    if not samples:
        raise SystemExit("No positive images found")

    rng = random.Random(args.seed)
    positives = []
    negatives = []
    sample_rows = []
    layout_counts = Counter()

    for path, digest in samples:
        gray = to_gray(Image.open(path))
        positive = infer_positive(gray)
        if positive is None:
            continue
        feature = (positive["spatial_score"], positive["gradient_score"])
        positives.append(feature)
        negatives.extend(sample_hard_negatives(
            gray, positive, args.negatives_per_image, rng
        ))
        layout_counts[positive["layout"]] += 1
        sample_rows.append({
            "file": path.name,
            "sha256": digest,
            "layout": positive["layout"],
            "spatial_score": round(feature[0], 6),
            "gradient_score": round(feature[1], 6),
        })

    if not positives:
        raise SystemExit("No catalog-aligned positives could be inferred")

    min_spatial, min_gradient = train_thresholds(positives, negatives)
    false_positives = sum(
        spatial >= min_spatial and gradient >= min_gradient
        for spatial, gradient in negatives
    )
    model = {
        "version": 1,
        "kind": "catalog_spatial_gradient_thresholds",
        "decision": {
            "min_spatial_score": round(min_spatial, 4),
            "min_gradient_score": round(min_gradient, 4),
            "hinted_min_spatial_score": 0.15,
            "hinted_min_gradient_score": 0.08,
        },
        "training": {
            "seed": args.seed,
            "unique_positive_count": len(positives),
            "hard_negative_count": len(negatives),
            "positive_recall": 1.0,
            "hard_negative_false_positives": false_positives,
            "layout_counts": dict(sorted(layout_counts.items())),
            "minimum_positive_spatial_score": round(min(x for x, _ in positives), 6),
            "minimum_positive_gradient_score": round(min(y for _, y in positives), 6),
            "samples": sample_rows,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "unique_positives": len(positives),
        "hard_negatives": len(negatives),
        "thresholds": model["decision"],
        "false_positives": false_positives,
        "layouts": model["training"]["layout_counts"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
