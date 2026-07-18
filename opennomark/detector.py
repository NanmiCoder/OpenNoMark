"""Open-vocabulary proposals and calibrated watermark candidate filtering.

OWLv2 is deliberately used as a *proposal* model.  The calibrated
``corner_signature`` lane preserves the high-recall handling of compact image
generator signatures.  A second, deliberately conservative
``generic_anywhere`` lane admits non-corner and square marks only when a
watermark-specific prompt is strong or multiple watermark-specific prompts
agree.  Plain object prompts such as ``logo`` and ``icon`` may expand a
supported region, but can never trigger removal by themselves.

Nothing in this module branches on a provider or filename.
"""

import json
import math
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
    CORNER_QUERY_LABELS = (
        "watermark",
        "brand watermark",
        "logo",
        "icon",
        "symbol",
        "badge",
        "stamp",
    )
    GENERIC_QUERY_LABELS = (
        "text watermark",
        "logo watermark",
        "copyright watermark",
        "transparent watermark",
    )
    QUERY_THRESHOLDS = {
        "watermark": 0.05,
        # This phrase recovers complete icon+text signatures when OWLv2 sees
        # only the icon for the shorter query.  Its lower threshold was
        # calibrated on the repository's real-image acceptance set.
        "brand watermark": 0.03,
        # These phrases participate only in the conservative generic lane.
        # They remain proposal thresholds rather than removal thresholds; the
        # generic acceptance gate below is intentionally much stricter.
        "text watermark": 0.05,
        "logo watermark": 0.05,
        "copyright watermark": 0.05,
        "transparent watermark": 0.05,
        "logo": 0.05,
        "icon": 0.05,
        "symbol": 0.05,
        "badge": 0.05,
        "stamp": 0.05,
    }
    CORNER_WATERMARK_LABELS = frozenset({"watermark", "brand watermark"})
    GENERIC_WATERMARK_LABELS = frozenset(
        {
            "watermark",
            "brand watermark",
            "text watermark",
            "logo watermark",
            "copyright watermark",
            "transparent watermark",
        }
    )
    # Backwards-compatible class attribute for code that inspected the former
    # two-label trusted set directly.
    TRUSTED_WATERMARK_LABELS = CORNER_WATERMARK_LABELS

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
        generic_geometry = profile.get("generic_anywhere", {})
        configured_queries = profile.get("query_thresholds", {})
        self.query_thresholds = {
            **self.QUERY_THRESHOLDS,
            **{
                key: float(value)
                for key, value in configured_queries.items()
                if key in self.QUERY_THRESHOLDS
            },
        }
        configured_trusted = profile.get("trusted_labels", self.CORNER_WATERMARK_LABELS)
        self.trusted_watermark_labels = frozenset(
            label for label in configured_trusted if label in self.query_thresholds
        ) or self.CORNER_WATERMARK_LABELS
        configured_generic_trusted = profile.get(
            "generic_trusted_labels", self.GENERIC_WATERMARK_LABELS
        )
        self.generic_watermark_labels = frozenset(
            label for label in configured_generic_trusted if label in self.query_thresholds
        ) or self.GENERIC_WATERMARK_LABELS
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
        self.containment_overlap = float(geometry.get("containment_overlap", 0.75))
        self.max_regions = max(1, int(geometry.get("max_regions", 4)))

        self.generic_single_min_score = float(
            generic_geometry.get("single_min_score", 0.20)
        )
        self.generic_agreement_min_score = float(
            generic_geometry.get("agreement_min_score", 0.10)
        )
        self.generic_min_supporting_prompts = max(
            2, int(generic_geometry.get("min_supporting_prompts", 2))
        )
        self.generic_min_width_ratio = float(
            generic_geometry.get("min_width_ratio", 0.02)
        )
        self.generic_max_width_ratio = float(
            generic_geometry.get("max_width_ratio", 0.60)
        )
        self.generic_min_height_ratio = float(
            generic_geometry.get("min_height_ratio", 0.01)
        )
        self.generic_max_height_ratio = float(
            generic_geometry.get("max_height_ratio", 0.35)
        )
        self.generic_max_area_ratio = float(
            generic_geometry.get("max_area_ratio", 0.12)
        )
        self.max_total_area_ratio = float(
            generic_geometry.get("max_total_area_ratio", 0.12)
        )
        self.last_filter_report = {}

        model_id = "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)

    def detect(self, image):
        """Detect watermark candidates in image. Returns list of {box, label, score}."""
        # Preserve the exact calibrated query competition used by the existing
        # platform corpus. Generic phrases run in a separate OWLv2 pass so
        # adding them cannot dilute or relabel the established corner signal.
        text_queries = [
            [label for label in self.CORNER_QUERY_LABELS if label in self.query_thresholds],
            [label for label in self.GENERIC_QUERY_LABELS if label in self.query_thresholds],
        ]

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
        """Accept calibrated corner signatures plus conservative generic marks.

        Every proposal is normalized and grouped into an overlap cluster
        first.  This makes prompt preference local to one visual mark, so a
        strong ``watermark`` proposal in one corner cannot suppress a distinct
        ``brand watermark`` elsewhere.  Accepted clusters are ordered with the
        calibrated corner lane first, then limited by count and total coarse
        box area before masks are ever constructed.
        """
        short_side = min(image_width, image_height)
        image_area = max(1.0, float(image_width * image_height))
        normalized = [
            candidate
            for item in boxes
            if (candidate := self._normalize_candidate(item, image_width, image_height))
            is not None
        ]
        clusters = self._cluster_candidates(normalized)
        accepted = []
        for cluster in clusters:
            region = self._accepted_cluster_region(
                cluster,
                image_width=image_width,
                image_height=image_height,
                short_side=short_side,
                image_area=image_area,
            )
            if region is not None:
                accepted.append(region)

        tier_order = {"corner_signature": 0, "generic_anywhere": 1}
        accepted.sort(
            key=lambda item: (
                tier_order[item["detection_tier"]],
                -float(item["score"]),
                -self._box_area(item["box"]),
                tuple(item["box"]),
                item["label"],
            )
        )
        corner_regions = [
            item for item in accepted if item["detection_tier"] == "corner_signature"
        ]
        generic_regions = [
            item for item in accepted if item["detection_tier"] == "generic_anywhere"
        ]
        corner_overflow = len(corner_regions) > self.max_regions
        selected = [] if corner_overflow else list(corner_regions)
        remaining_capacity = max(0, self.max_regions - len(selected))
        area_budget = image_area * self.max_total_area_ratio
        generic_area = sum(self._box_area(item["box"]) for item in generic_regions)
        overflow_reasons = []
        if len(generic_regions) > remaining_capacity:
            overflow_reasons.append("max_regions")
        if generic_area > area_budget + 1e-6:
            overflow_reasons.append("area_budget")
        # A tiled or otherwise excessive generic detection set must fail
        # closed. Removing a convenient prefix would produce a partially
        # cleaned image that downstream residual checks could misclassify as
        # complete.
        if not corner_overflow and not overflow_reasons:
            selected.extend(generic_regions)

        self.last_filter_report = {
            "candidate_clusters": int(len(clusters)),
            "eligible_corner_regions": int(len(corner_regions)),
            "accepted_corner_regions": int(0 if corner_overflow else len(corner_regions)),
            "corner_regions_truncated": bool(corner_overflow),
            "eligible_generic_regions": int(len(generic_regions)),
            "accepted_generic_regions": int(0 if overflow_reasons else len(generic_regions)),
            "generic_overflow": bool(overflow_reasons),
            "overflow_reason": "+".join(overflow_reasons) if overflow_reasons else None,
            "max_regions": int(self.max_regions),
            "max_total_area_ratio": float(self.max_total_area_ratio),
            "generic_area_ratio": float(generic_area / image_area),
        }
        return selected

    def _normalize_candidate(self, item, image_width, image_height):
        """Return a bounded, JSON-safe proposal or ``None`` when malformed."""
        label = item.get("label")
        box = item.get("box")
        if label not in self.query_thresholds or not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        try:
            raw_score = float(item.get("score", 0.0))
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(raw_score) or not all(
            math.isfinite(value) for value in (x1, y1, x2, y2)
        ):
            return None
        if raw_score < self.query_thresholds[label]:
            return None

        normalized_box = [
            max(0.0, min(float(image_width), x1)),
            max(0.0, min(float(image_height), y1)),
            max(0.0, min(float(image_width), x2)),
            max(0.0, min(float(image_height), y2)),
        ]
        if normalized_box[2] <= normalized_box[0] or normalized_box[3] <= normalized_box[1]:
            return None
        return {
            "box": normalized_box,
            "label": str(label),
            "score": raw_score,
            "raw_score": raw_score,
        }

    def _cluster_candidates(self, candidates):
        """Group prompt variants that describe the same visual region."""
        ranked = sorted(
            candidates,
            key=lambda item: (
                -float(item["raw_score"]),
                -self._box_area(item["box"]),
                tuple(item["box"]),
                item["label"],
            ),
        )
        clusters = []
        for candidate in ranked:
            matching = [
                index
                for index, cluster in enumerate(clusters)
                if any(self._same_region(candidate["box"], item["box"]) for item in cluster)
            ]
            if not matching:
                clusters.append([candidate])
                continue

            primary_index = matching[0]
            clusters[primary_index].append(candidate)
            # Merge transitive overlap groups deterministically.
            for index in reversed(matching[1:]):
                clusters[primary_index].extend(clusters.pop(index))
        return clusters

    def _accepted_cluster_region(
        self,
        cluster,
        *,
        image_width,
        image_height,
        short_side,
        image_area,
    ):
        """Validate one cluster and choose its evidence-backed removal box."""
        watermark_candidates = [
            item for item in cluster if item["label"] in self.generic_watermark_labels
        ]
        if not watermark_candidates:
            return None

        corner_candidates = [
            item
            for item in watermark_candidates
            if item["label"] in self.trusted_watermark_labels
            and self._fits_corner_signature(
                item,
                image_width=image_width,
                image_height=image_height,
                short_side=short_side,
                image_area=image_area,
            )
        ]
        generic_candidates = [
            item
            for item in watermark_candidates
            if self._fits_generic_geometry(item, short_side=short_side, image_area=image_area)
        ]

        if corner_candidates:
            detection_tier = "corner_signature"
            # A corner candidate necessarily also fits the looser generic
            # shape envelope. Restricting evidence to that envelope prevents a
            # huge overlapping proposal from replacing a valid compact box.
            eligible_evidence = generic_candidates or corner_candidates
        elif generic_candidates:
            best_generic_score = max(float(item["raw_score"]) for item in generic_candidates)
            supporting_prompts = {item["label"] for item in generic_candidates}
            strong_single = best_generic_score >= self.generic_single_min_score
            prompt_agreement = (
                best_generic_score >= self.generic_agreement_min_score
                and len(supporting_prompts) >= self.generic_min_supporting_prompts
            )
            if not (strong_single or prompt_agreement):
                return None
            detection_tier = "generic_anywhere"
            eligible_evidence = generic_candidates
        else:
            return None

        primary = min(
            eligible_evidence,
            key=lambda item: (
                -float(item["raw_score"]),
                -self._box_area(item["box"]),
                tuple(item["box"]),
                item["label"],
            ),
        )
        best_score = float(primary["raw_score"])
        expansion_candidates = [
            item
            for item in cluster
            # Clusters are transitive: a chain of overlapping proposals may
            # contain two boxes that do not describe the same visual mark.
            # Expansion must therefore stay directly attached to the primary
            # proposal, otherwise a valid corner signature can inherit an
            # unrelated scene-text box from the far end of the chain.
            if self._same_region(primary["box"], item["box"])
            and self._fits_generic_geometry(item, short_side=short_side, image_area=image_area)
            and self._box_area(item["box"]) > self._box_area(primary["box"])
            and float(item["raw_score"]) >= best_score * 0.5
        ]
        expansion = max(
            expansion_candidates,
            key=lambda item: (
                self._box_area(item["box"]),
                float(item["raw_score"]),
                tuple(item["box"]),
            ),
            default=primary,
        )
        supporting_labels = sorted({item["label"] for item in cluster})
        region = {
            "box": [float(value) for value in expansion["box"]],
            "label": primary["label"],
            "score": best_score,
            "raw_score": best_score,
            "detection_tier": detection_tier,
            "supporting_labels": supporting_labels,
        }
        if expansion["label"] != primary["label"]:
            region["box_source_label"] = expansion["label"]
        return region

    def _fits_corner_signature(
        self, item, *, image_width, image_height, short_side, image_area
    ):
        x1, y1, x2, y2 = item["box"]
        width = x2 - x1
        height = y2 - y1
        width_ratio = width / short_side
        height_ratio = height / short_side
        if not (
            self.min_width_ratio <= width_ratio <= self.max_width_ratio
            and self.min_height_ratio <= height_ratio <= self.max_height_ratio
            and width / height >= self.min_aspect_ratio
            and (width * height) / image_area <= self.max_area_ratio
        ):
            return False

        horizontal_gap = min(max(0.0, x1), max(0.0, image_width - x2)) / image_width
        vertical_gap = min(max(0.0, y1), max(0.0, image_height - y2)) / image_height
        if horizontal_gap > self.edge_ratio or vertical_gap > self.edge_ratio:
            return False

        nearest_vertical_edge = "top" if y1 <= image_height - y2 else "bottom"
        return not (
            nearest_vertical_edge == "top"
            and float(item["raw_score"]) < self.top_edge_min_score
        )

    def _fits_generic_geometry(self, item, *, short_side, image_area):
        x1, y1, x2, y2 = item["box"]
        width = x2 - x1
        height = y2 - y1
        return (
            self.generic_min_width_ratio <= width / short_side <= self.generic_max_width_ratio
            and self.generic_min_height_ratio
            <= height / short_side
            <= self.generic_max_height_ratio
            and (width * height) / image_area <= self.generic_max_area_ratio
        )

    def _same_region(self, first, second):
        return (
            self._iou(first, second) >= self.nms_iou
            or self._intersection_over_smaller(first, second) >= self.containment_overlap
        )

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
