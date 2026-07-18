"""Provider-agnostic watermark localization facade.

The production pipeline consumes one region contract regardless of how visual
evidence was found.  High-precision shape evidence and open-vocabulary text
evidence are implementation details of this module; filenames and provider
directories never participate in routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image, ImageDraw, ImageFilter

from .gemini_alpha import create_gemini_mask, detect_gemini_watermark


@dataclass
class LocalizedWatermark:
    """An internal region plus its precise removal mask."""

    box: list[float]
    score: float
    source: str
    method: str
    mask: Image.Image
    details: dict[str, Any] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, Any]:
        metadata = {
            "box": [float(value) for value in self.box],
            "score": float(self.score),
            "source": self.source,
            "method": self.method,
        }
        mask_box = self.mask.getbbox()
        if mask_box is not None:
            metadata["mask_box"] = [float(value) for value in mask_box]
        if self.details:
            metadata["details"] = self.details
        return metadata


class WatermarkLocalizer:
    """Fuse visual experts behind a single, stable localization API."""

    GENERIC_SEMANTIC_ONLY_MIN_SCORE = 0.60

    def __init__(
        self,
        device=None,
        detector_factory: Callable | None = None,
        text_detector_factory: Callable | None = None,
    ):
        self.device = device
        self._detector = None
        self._detector_factory = detector_factory
        self._text_detector = None
        self._text_detector_factory = text_detector_factory

    @property
    def detector(self):
        if self._detector is None:
            if self._detector_factory is not None:
                self._detector = self._detector_factory()
            else:
                from .detector import WatermarkDetector

                self._detector = WatermarkDetector(device=self.device)
        return self._detector

    @property
    def text_detector(self):
        if self._text_detector is None:
            if self._text_detector_factory is not None:
                self._text_detector = self._text_detector_factory()
            else:
                from .text_detector import TextWatermarkDetector

                self._text_detector = TextWatermarkDetector(device=self.device)
        return self._text_detector

    def localize(self, image: Image.Image) -> tuple[list[LocalizedWatermark], dict[str, Any]]:
        """Return accepted regions and serializable localization evidence."""
        gemini = detect_gemini_watermark(image)
        raw = self.detector.detect(image)
        accepted = self.detector.filter_watermarks(raw, image.width, image.height)
        semantic_regions = [self._region_from_box(image.size, item) for item in accepted]
        filter_report = dict(getattr(self.detector, "last_filter_report", {}) or {})

        # Known platform evidence stays on the fast path. OCR is a fallback for
        # images without that evidence, and a mask refiner for generic semantic
        # detections. This keeps the established 80-image platform path fast
        # while adding textual and non-corner coverage.
        generic_semantic = any(
            item.get("detection_tier") == "generic_anywhere" for item in accepted
        )
        known_semantic = bool(accepted) and not generic_semantic
        should_run_text = generic_semantic or (not gemini.get("found") and not known_semantic)
        text_raw = []
        text_regions: list[LocalizedWatermark] = []
        text_report: dict[str, Any] = {}
        if should_run_text:
            text_raw = self.text_detector.detect(image)
            text_accepted = self.text_detector.filter_watermarks(
                text_raw, image.width, image.height
            )
            text_regions = [
                self._region_from_text(image.size, item) for item in text_accepted
            ]
            text_report = dict(
                getattr(self.text_detector, "last_filter_report", {}) or {}
            )

        corner_semantic = [
            region
            for region in semantic_regions
            if region.details.get("detector_profile") != "generic_anywhere"
        ]
        generic_semantic_regions = [
            region
            for region in semantic_regions
            if region.details.get("detector_profile") == "generic_anywhere"
        ]
        known_platform_evidence = bool(gemini.get("found") or corner_semantic)
        if known_platform_evidence:
            # When a known platform mark already exists, OCR may refine or
            # validate a generic proposal but must not independently erase
            # unrelated scene text elsewhere in the image.
            text_regions = [
                text_region
                for text_region in text_regions
                if any(
                    self._intersection_over_smaller(text_region.box, semantic.box)
                    >= 0.25
                    for semantic in generic_semantic_regions
                )
            ]

        confirmed_generic = [
            region
            for region in generic_semantic_regions
            if region.score >= self.GENERIC_SEMANTIC_ONLY_MIN_SCORE
            or any(
                self._intersection_over_smaller(region.box, text_region.box) >= 0.25
                for text_region in text_regions
            )
        ]
        suppressed_generic_count = len(generic_semantic_regions) - len(confirmed_generic)
        semantic_regions = [*corner_semantic, *confirmed_generic]

        if gemini.get("found"):
            spatial_region = self._region_from_spatial_template(image.size, gemini)
            regions, arbitration, decisions = self._fuse_spatial_and_semantic(
                image.size,
                spatial_region,
                semantic_regions,
            )
            regions = self._fuse_text_regions(regions, text_regions)
            evidence = {
                "total_proposals": len(raw) + len(text_raw) + 1,
                "accepted_regions": len(regions),
                "experts": ["spatial_template", "open_vocabulary"],
                "arbitration": arbitration,
            }
            if should_run_text:
                evidence["experts"].append("ocr_text")
            if len(decisions) > 1:
                evidence["arbitration_decisions"] = decisions
            if suppressed_generic_count:
                evidence["suppressed_generic_regions"] = suppressed_generic_count
            self._record_overflow(evidence, filter_report, text_report)
            return regions, evidence

        regions = self._fuse_text_regions(semantic_regions, text_regions)
        evidence = {
            "total_proposals": len(raw) + len(text_raw),
            "accepted_regions": len(regions),
            "experts": ["open_vocabulary"] + (["ocr_text"] if should_run_text else []),
        }
        if suppressed_generic_count:
            evidence["suppressed_generic_regions"] = suppressed_generic_count
        self._record_overflow(evidence, filter_report, text_report)
        return regions, evidence

    @staticmethod
    def _record_overflow(evidence, semantic_report, text_report):
        overflow = []
        if semantic_report.get("corner_regions_truncated"):
            overflow.append(
                {
                    "expert": "open_vocabulary",
                    "reason": "max_regions",
                }
            )
        if semantic_report.get("generic_overflow"):
            overflow.append(
                {
                    "expert": "open_vocabulary",
                    "reason": semantic_report.get("overflow_reason") or "candidate_budget",
                }
            )
        if text_report.get("overflow"):
            overflow.append(
                {
                    "expert": "ocr_text",
                    "reason": "+".join(text_report.get("reasons", []))
                    or "candidate_budget",
                }
            )
        if overflow:
            evidence["safety"] = {
                "automatic_removal_blocked": True,
                "overflow": overflow,
            }

    @classmethod
    def _fuse_text_regions(
        cls,
        existing_regions: list[LocalizedWatermark],
        text_regions: list[LocalizedWatermark],
    ) -> list[LocalizedWatermark]:
        """Fuse OCR polygons without weakening known-platform mask priority."""
        fused = list(existing_regions)
        for text_region in text_regions:
            overlaps = [
                index
                for index, existing in enumerate(fused)
                if cls._intersection_over_smaller(existing.box, text_region.box) >= 0.25
            ]
            if not overlaps:
                fused.append(text_region)
                continue

            primary_index = overlaps[0]
            primary = fused[primary_index]
            profile = primary.details.get("detector_profile")
            if primary.source == "spatial_template" or profile == "corner_signature":
                validation_sources = set(
                    primary.details.get("validation_sources", [primary.source])
                )
                validation_sources.add("ocr_text")
                primary.details["validation_sources"] = sorted(validation_sources)
                primary.details["supporting_text"] = text_region.details.get("text", "")
            else:
                # A recognized text polygon is tighter than a generic OWLv2
                # rectangle. Keep the polygon for LaMa and preserve semantic
                # evidence for residual validation.
                validation_sources = set(
                    text_region.details.get("validation_sources", ["ocr_text"])
                )
                validation_sources.add(primary.source)
                text_region.details["validation_sources"] = sorted(validation_sources)
                text_region.details["supporting_semantic_label"] = primary.details.get(
                    "label", "watermark"
                )
                fused[primary_index] = text_region

            for index in reversed(overlaps[1:]):
                fused.pop(index)

        priority = {
            "spatial_template": 0,
            "open_vocabulary": 1,
            "ocr_text": 2,
        }
        return sorted(
            fused,
            key=lambda region: (
                priority.get(region.source, 9),
                0
                if region.details.get("detector_profile") == "corner_signature"
                else 1,
                -float(region.score),
                tuple(float(value) for value in region.box),
            ),
        )

    @classmethod
    def _fuse_spatial_and_semantic(
        cls,
        image_size: tuple[int, int],
        spatial: LocalizedWatermark,
        semantic_regions: list[LocalizedWatermark],
    ) -> tuple[list[LocalizedWatermark], str, list[dict[str, Any]]]:
        """Prefer the precise expert for one mark and keep other real regions.

        The old facade chose exactly one candidate whenever the Gemini template
        fired.  That protected its shape mask, but it also discarded a second,
        unrelated watermark.  Fusion now remains conservative per overlap
        cluster: the precise mask wins duplicate evidence, the existing
        same-corner false-positive guard can still replace it, and independent
        semantic regions survive as additional removal targets.
        """
        if not semantic_regions:
            return [spatial], "spatial_only", []

        decisions = []
        override = None
        primary_summary = None
        for semantic in semantic_regions:
            winner, decision = cls._arbitrate_spatial_and_semantic(
                image_size,
                spatial,
                [semantic],
            )
            decisions.append(
                {
                    "decision": decision,
                    "semantic_box": [float(value) for value in semantic.box],
                }
            )
            if primary_summary is None or decision != "spatial_different_corner":
                primary_summary = decision
            if winner is semantic and decision == "semantic_override":
                override = semantic
                primary_summary = decision
                break

        if override is not None:
            override.details["validation_sources"] = [
                "open_vocabulary",
                "spatial_template",
            ]
            kept = [override]
            for semantic in semantic_regions:
                if semantic is override or any(
                    cls._intersection_over_smaller(semantic.box, item.box) >= 0.75
                    for item in kept
                ):
                    continue
                kept.append(semantic)
            return kept, "semantic_override", decisions

        duplicates = [
            semantic
            for semantic in semantic_regions
            if cls._intersection_over_smaller(spatial.box, semantic.box) >= 0.25
        ]
        if duplicates:
            spatial.details["validation_sources"] = [
                "spatial_template",
                "open_vocabulary",
            ]
            spatial.details["supporting_labels"] = sorted(
                {
                    label
                    for semantic in duplicates
                    for label in (
                        semantic.details.get("supporting_labels")
                        or [semantic.details.get("label", "watermark")]
                    )
                }
            )

        kept = [spatial]
        for semantic in semantic_regions:
            if semantic in duplicates:
                continue
            # ``corner_signature`` is the calibrated fallback for images that
            # do not have a precise platform template. Once the shape expert
            # has fired, a non-overlapping corner proposal that lost the
            # arbitration is more likely to be scene text than a second mark.
            # A separately confirmed ``generic_anywhere`` region still
            # survives, which preserves intentional multi-watermark support.
            if semantic.details.get("detector_profile") == "corner_signature":
                continue
            if any(
                cls._intersection_over_smaller(semantic.box, item.box) >= 0.75
                for item in kept
            ):
                continue
            kept.append(semantic)
        return kept, primary_summary or "spatial_only", decisions

    @classmethod
    def _arbitrate_spatial_and_semantic(
        cls,
        image_size: tuple[int, int],
        spatial: LocalizedWatermark,
        semantic_regions: list[LocalizedWatermark],
    ) -> tuple[LocalizedWatermark, str]:
        """Resolve rare catalog-template false positives without provider hints.

        A real Gemini sparkle has a precise shape mask, so it remains the
        default whenever the experts agree, disagree on the corner, or the
        semantic evidence is weaker.  A non-overlapping text signature in the
        same corner may replace it only when OWLv2 scores it at least as
        strongly and places it materially closer to the image edges.  This
        catches catalog-shaped background texture while preserving difficult,
        low-confidence Gemini positives from the calibrated corpus.
        """
        if not semantic_regions:
            return spatial, "spatial_only"

        semantic = semantic_regions[0]
        if cls._corner(spatial.box, image_size) != cls._corner(semantic.box, image_size):
            return spatial, "spatial_different_corner"
        if cls._intersection_over_smaller(spatial.box, semantic.box) >= 0.25:
            return spatial, "spatial_overlapping_evidence"
        if semantic.score < spatial.score:
            return spatial, "spatial_stronger_score"
        if cls._edge_distance(semantic.box, image_size) >= cls._edge_distance(
            spatial.box, image_size
        ):
            return spatial, "spatial_closer_to_edges"

        semantic.details["suppressed_spatial_score"] = float(spatial.score)
        return semantic, "semantic_override"

    @staticmethod
    def _corner(box: list[float], image_size: tuple[int, int]) -> tuple[str, str]:
        width, height = image_size
        center_x = (float(box[0]) + float(box[2])) / 2.0
        center_y = (float(box[1]) + float(box[3])) / 2.0
        return (
            "left" if center_x < width / 2.0 else "right",
            "top" if center_y < height / 2.0 else "bottom",
        )

    @staticmethod
    def _edge_distance(box: list[float], image_size: tuple[int, int]) -> float:
        width, height = image_size
        horizontal = min(max(0.0, float(box[0])), max(0.0, width - float(box[2])))
        vertical = min(max(0.0, float(box[1])), max(0.0, height - float(box[3])))
        return horizontal / max(1, width) + vertical / max(1, height)

    @staticmethod
    def _intersection_over_smaller(first: list[float], second: list[float]) -> float:
        x1 = max(float(first[0]), float(second[0]))
        y1 = max(float(first[1]), float(second[1]))
        x2 = min(float(first[2]), float(second[2]))
        y2 = min(float(first[3]), float(second[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, float(first[2]) - float(first[0])) * max(
            0.0, float(first[3]) - float(first[1])
        )
        second_area = max(0.0, float(second[2]) - float(second[0])) * max(
            0.0, float(second[3]) - float(second[1])
        )
        smaller = min(first_area, second_area)
        return intersection / smaller if smaller else 0.0

    def localize_residuals(
        self,
        image: Image.Image,
        original_regions: list[LocalizedWatermark],
    ) -> list[LocalizedWatermark]:
        """Re-run only the experts that produced the original regions."""
        sources = {region.source for region in original_regions}
        for region in original_regions:
            sources.update(region.details.get("validation_sources", []))

        spatial = None
        if "spatial_template" in sources:
            detection = detect_gemini_watermark(image)
            if detection.get("found"):
                spatial = self._region_from_spatial_template(image.size, detection)

        semantic_regions: list[LocalizedWatermark] = []
        if "open_vocabulary" in sources:
            raw = self.detector.detect(image)
            accepted = self.detector.filter_watermarks(raw, image.width, image.height)
            semantic_regions = [self._region_from_box(image.size, item) for item in accepted]

        text_regions: list[LocalizedWatermark] = []
        if "ocr_text" in sources:
            proposals = self.text_detector.detect(image)
            accepted = self.text_detector.filter_watermarks(
                proposals, image.width, image.height
            )
            text_regions = [
                self._region_from_text(image.size, item) for item in accepted
            ]

        if spatial is not None and semantic_regions:
            residuals, _, _ = self._fuse_spatial_and_semantic(
                image.size,
                spatial,
                semantic_regions,
            )
        elif spatial is not None:
            residuals = [spatial]
        else:
            residuals = semantic_regions
        return self._fuse_text_regions(residuals, text_regions)

    @staticmethod
    def _region_from_spatial_template(image_size, detection) -> LocalizedWatermark:
        x = float(detection["x"])
        y = float(detection["y"])
        size = float(detection["logo_size"])
        details = {
            "layout": detection["layout"],
            "spatial_score": float(detection["spatial_score"]),
            "gradient_score": float(detection["gradient_score"]),
            "decision": detection["decision"],
            "model_version": detection.get("model_version", 1),
        }
        return LocalizedWatermark(
            box=[x, y, x + size, y + size],
            score=float(detection["confidence"]),
            source="spatial_template",
            method="shape_mask",
            mask=create_gemini_mask(image_size, detection),
            details=details,
        )

    @staticmethod
    def _region_from_box(image_size, item) -> LocalizedWatermark:
        # Import lazily to keep localization tests independent from LaMa.
        from .inpainter import create_box_mask

        # OWLv2 boxes follow the high-contrast glyph interior and can miss the
        # translucent antialiasing fringe by several pixels.  Scale this small
        # safety margin with resolution; it is still far tighter than the old
        # large rectangular masks that caused structure bleed.
        padding = max(6, int(round(min(image_size) * 0.006)))
        details = {
            "label": item.get("label", "watermark"),
            "raw_score": float(item.get("raw_score", item.get("score", 0.0))),
            "mask_padding": padding,
        }
        detection_tier = item.get("detection_tier")
        if detection_tier:
            details["detector_profile"] = detection_tier
        supporting_labels = item.get("supporting_labels")
        if supporting_labels:
            details["supporting_labels"] = list(supporting_labels)
        return LocalizedWatermark(
            box=[float(value) for value in item["box"]],
            score=float(item.get("score", 0.0)),
            source="open_vocabulary",
            method="box_mask",
            mask=create_box_mask(image_size, [item], padding=padding),
            details=details,
        )

    @staticmethod
    def _region_from_text(image_size, item) -> LocalizedWatermark:
        polygon = [
            (float(point[0]), float(point[1]))
            for point in item.get("polygon", [])
            if len(point) >= 2
        ]
        if len(polygon) < 3:
            x1, y1, x2, y2 = (float(value) for value in item["box"])
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        mask = Image.new("L", image_size, 0)
        ImageDraw.Draw(mask).polygon(polygon, fill=255)
        padding = max(3, int(round(min(image_size) * 0.003)))
        filter_size = padding * 2 + 1
        mask = mask.filter(ImageFilter.MaxFilter(filter_size))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2))
        detection_score = float(item.get("detection_score", 0.0))
        recognition_score = float(item.get("recognition_score", 0.0))
        details = {
            "text": str(item.get("text", "")),
            "detection_score": detection_score,
            "recognition_score": recognition_score,
            "evidence": list(item.get("evidence", [])),
            "detector_profile": item.get("detection_tier", "generic_ocr"),
            "mask_padding": padding,
        }
        return LocalizedWatermark(
            box=[float(value) for value in item["box"]],
            score=min(detection_score, recognition_score),
            source="ocr_text",
            method="polygon_mask",
            mask=mask,
            details=details,
        )
