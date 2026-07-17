"""Provider-agnostic watermark localization facade.

The production pipeline consumes one region contract regardless of how visual
evidence was found.  High-precision shape evidence and open-vocabulary text
evidence are implementation details of this module; filenames and provider
directories never participate in routing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

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

    def __init__(self, device=None, detector_factory: Callable | None = None):
        self.device = device
        self._detector = None
        self._detector_factory = detector_factory

    @property
    def detector(self):
        if self._detector is None:
            if self._detector_factory is not None:
                self._detector = self._detector_factory()
            else:
                from .detector import WatermarkDetector

                self._detector = WatermarkDetector(device=self.device)
        return self._detector

    def localize(self, image: Image.Image) -> tuple[list[LocalizedWatermark], dict[str, Any]]:
        """Return accepted regions and serializable localization evidence."""
        gemini = detect_gemini_watermark(image)
        if gemini.get("found"):
            spatial_region = self._region_from_spatial_template(image.size, gemini)
            raw = self.detector.detect(image)
            accepted = self.detector.filter_watermarks(raw, image.width, image.height)
            semantic_regions = [self._region_from_box(image.size, item) for item in accepted]
            region, arbitration = self._arbitrate_spatial_and_semantic(
                image.size,
                spatial_region,
                semantic_regions,
            )
            return [region], {
                "total_proposals": len(raw) + 1,
                "accepted_regions": 1,
                "experts": ["spatial_template", "open_vocabulary"],
                "arbitration": arbitration,
            }

        raw = self.detector.detect(image)
        accepted = self.detector.filter_watermarks(raw, image.width, image.height)
        regions = [self._region_from_box(image.size, item) for item in accepted]
        return regions, {
            "total_proposals": len(raw),
            "accepted_regions": len(regions),
            "experts": ["open_vocabulary"],
        }

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
        residuals: list[LocalizedWatermark] = []
        if "spatial_template" in sources:
            detection = detect_gemini_watermark(image)
            if detection.get("found"):
                residuals.append(self._region_from_spatial_template(image.size, detection))
        if "open_vocabulary" in sources:
            raw = self.detector.detect(image)
            accepted = self.detector.filter_watermarks(raw, image.width, image.height)
            residuals.extend(self._region_from_box(image.size, item) for item in accepted)
        return residuals

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
