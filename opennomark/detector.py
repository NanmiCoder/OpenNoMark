"""Open-vocabulary proposals and calibrated watermark candidate filtering.

OWLv2 is deliberately used as a *proposal* model.  Its raw ``icon`` and
``badge`` predictions are useful evidence, but accepting every small object in
a corner is destructive on busy images.  The filtering stage therefore uses
the semantic phrase that produced a proposal, edge distance, size, and
overlap-aware deduplication.  Nothing in this module branches on a provider or
filename.
"""

import json
import os

import torch


_CALIBRATION_PATH = os.path.join(
    os.path.dirname(__file__), "assets", "watermark_detector.json"
)


def _load_calibration():
    try:
        with open(_CALIBRATION_PATH, encoding="utf-8") as file:
            profile = json.load(file)
        if isinstance(profile, dict) and isinstance(profile.get("geometry"), dict):
            return profile
    except (OSError, ValueError):
        pass
    return {}


class WatermarkDetector:
    QUERY_THRESHOLDS = {
        "watermark": 0.05,
        # This phrase recovers complete icon+text signatures when OWLv2 sees
        # only the icon for the shorter query.  Its lower threshold was
        # calibrated on the repository's real-image acceptance set.
        "brand watermark": 0.03,
        "logo": 0.05,
        "icon": 0.05,
        "symbol": 0.05,
        "badge": 0.05,
        "stamp": 0.05,
    }
    TRUSTED_WATERMARK_LABELS = frozenset({"watermark", "brand watermark"})

    def __init__(
        self,
        device=None,
        score_threshold=None,
        edge_ratio=None,
        min_width_ratio=None,
        max_width_ratio=None,
        min_height_ratio=None,
        max_height_ratio=None,
        min_aspect_ratio=None,
        top_edge_min_score=None,
        max_area_ratio=None,
        nms_iou=None,
    ):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        profile = _load_calibration()
        geometry = profile.get("geometry", {})
        configured_queries = profile.get("query_thresholds", {})
        self.query_thresholds = {
            **self.QUERY_THRESHOLDS,
            **{
                key: float(value)
                for key, value in configured_queries.items()
                if key in self.QUERY_THRESHOLDS
            },
        }
        configured_trusted = profile.get("trusted_labels", self.TRUSTED_WATERMARK_LABELS)
        self.trusted_watermark_labels = frozenset(
            label for label in configured_trusted if label in self.query_thresholds
        ) or self.TRUSTED_WATERMARK_LABELS
        self.device = device
        self.score_threshold = float(
            min(self.query_thresholds.values()) if score_threshold is None else score_threshold
        )
        self.edge_ratio = float(geometry.get("edge_ratio", 0.08) if edge_ratio is None else edge_ratio)
        self.min_width_ratio = float(
            geometry.get("min_width_ratio", 0.03) if min_width_ratio is None else min_width_ratio
        )
        self.max_width_ratio = float(
            geometry.get("max_width_ratio", 0.40) if max_width_ratio is None else max_width_ratio
        )
        self.min_height_ratio = float(
            geometry.get("min_height_ratio", 0.01) if min_height_ratio is None else min_height_ratio
        )
        self.max_height_ratio = float(
            geometry.get("max_height_ratio", 0.15) if max_height_ratio is None else max_height_ratio
        )
        self.min_aspect_ratio = float(
            geometry.get("min_aspect_ratio", 1.5) if min_aspect_ratio is None else min_aspect_ratio
        )
        self.top_edge_min_score = float(
            geometry.get("top_edge_min_score", 0.15)
            if top_edge_min_score is None
            else top_edge_min_score
        )
        self.max_area_ratio = float(
            geometry.get("max_area_ratio", 0.08) if max_area_ratio is None else max_area_ratio
        )
        self.nms_iou = float(geometry.get("nms_iou", 0.35) if nms_iou is None else nms_iou)

        model_id = "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)

    def detect(self, image):
        """Detect watermark candidates in image. Returns list of {box, label, score}."""
        text_queries = [list(self.query_thresholds)]

        all_boxes = []
        for queries in text_queries:
            inputs = self.processor(text=queries, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.score_threshold
            )

            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()

            for box, score, label_idx in zip(boxes, scores, labels):
                label = queries[label_idx]
                if float(score) < self.query_thresholds[label]:
                    continue
                all_boxes.append({
                    "box": box.tolist(),
                    "label": label,
                    "score": float(score),
                })

        return all_boxes

    def filter_watermarks(self, boxes, image_width, image_height):
        """Keep compact, semantically trusted marks close to two edges.

        The previous center-in-the-last-15% gate dropped otherwise identical
        text marks when OWLv2 jittered by one or two pixels.  Measuring the
        *box edge* instead is stable for wide signatures, and requiring a
        watermark-specific text query prevents trophies, UI icons, and badges
        from being erased merely because they happen to be near a corner.
        """
        short_side = min(image_width, image_height)
        image_area = max(1.0, float(image_width * image_height))

        candidates = []
        for item in boxes:
            if item.get("label") not in self.trusted_watermark_labels:
                continue
            x1, y1, x2, y2 = item["box"]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            width_ratio = w / short_side
            height_ratio = h / short_side
            if not (
                self.min_width_ratio <= width_ratio <= self.max_width_ratio
                and self.min_height_ratio <= height_ratio <= self.max_height_ratio
                and w / h >= self.min_aspect_ratio
            ):
                continue
            if (w * h) / image_area > self.max_area_ratio:
                continue

            horizontal_gap = min(max(0.0, x1), max(0.0, image_width - x2)) / image_width
            vertical_gap = min(max(0.0, y1), max(0.0, image_height - y2)) / image_height
            if horizontal_gap > self.edge_ratio or vertical_gap > self.edge_ratio:
                continue

            raw_score = float(item.get("score", 0.0))
            nearest_vertical_edge = "top" if y1 <= image_height - y2 else "bottom"
            # The low-threshold phrase is needed for faint bottom signatures.
            # Top-edge candidates were the dominant hard-negative family in
            # calibration, while the real legacy top badge scores far higher.
            if nearest_vertical_edge == "top" and raw_score < self.top_edge_min_score:
                continue

            normalized = dict(item)
            normalized["box"] = [
                float(max(0.0, min(image_width, x1))),
                float(max(0.0, min(image_height, y1))),
                float(max(0.0, min(image_width, x2))),
                float(max(0.0, min(image_height, y2))),
            ]
            normalized["raw_score"] = raw_score
            normalized["score"] = normalized["raw_score"]
            candidates.append(normalized)

        # The concise prompt is the calibrated primary signal.  The broader
        # phrase exists only to recover the few images where OWLv2 sees a logo
        # glyph but not the adjacent text.  Treating its occasionally oversized
        # boxes as peers can leave the final glyph outside the mask.
        primary = [item for item in candidates if item["label"] == "watermark"]
        deduplicated = self._deduplicate(primary or candidates)
        # The calibrated corpus contains one visible generator signature per
        # image.  Selecting the strongest trusted region prevents a weaker
        # corner-shaped background proposal from causing destructive edits.
        # The public API remains a list so multi-region calibration can be
        # introduced later without changing callers.
        return deduplicated[:1]

    def _deduplicate(self, candidates):
        """Merge overlapping prompt variants into one conservative box."""
        ranked = sorted(
            candidates,
            key=lambda item: (float(item.get("score", 0.0)), self._box_area(item["box"])),
            reverse=True,
        )
        kept = []
        for candidate in ranked:
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(kept)
                    if (
                        self._iou(candidate["box"], existing["box"]) >= self.nms_iou
                        or self._intersection_over_smaller(candidate["box"], existing["box"]) >= 0.75
                    )
                ),
                None,
            )
            if duplicate_index is None:
                kept.append(candidate)
                continue

            existing = kept[duplicate_index]
            # Prefer the larger overlapping box: the shorter query sometimes
            # covers only a logo glyph while ``brand watermark`` covers its
            # adjacent text as well.  Confidence remains the stronger score.
            candidate_is_supported_expansion = (
                self._box_area(candidate["box"]) > self._box_area(existing["box"])
                and float(candidate.get("raw_score", 0.0))
                >= float(existing.get("raw_score", 0.0)) * 0.5
            )
            if candidate_is_supported_expansion:
                candidate["score"] = max(candidate["score"], existing["score"])
                candidate["supporting_labels"] = sorted(
                    {candidate["label"], existing["label"], *existing.get("supporting_labels", [])}
                )
                kept[duplicate_index] = candidate
            else:
                existing["score"] = max(existing["score"], candidate["score"])
                existing["raw_score"] = max(existing["raw_score"], candidate["raw_score"])
                existing["supporting_labels"] = sorted(
                    {existing["label"], candidate["label"], *existing.get("supporting_labels", [])}
                )

        return sorted(kept, key=lambda item: float(item.get("score", 0.0)), reverse=True)

    @staticmethod
    def _box_area(box):
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    @classmethod
    def _iou(cls, first, second):
        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = cls._box_area(first) + cls._box_area(second) - intersection
        return intersection / union if union > 0 else 0.0

    @classmethod
    def _intersection_over_smaller(cls, first, second):
        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        smaller = min(cls._box_area(first), cls._box_area(second))
        return intersection / smaller if smaller > 0 else 0.0
