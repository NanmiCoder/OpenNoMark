"""Conservative OCR proposals for visible text watermarks.

This module is intentionally independent from the main localizer.  PP-OCRv5
provides text polygons and recognition evidence; lexical and geometry gates
then decide whether those proposals are safe enough for automatic removal.
The models are loaded only when :meth:`TextWatermarkDetector.detect` is first
called.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from PIL import Image


DEFAULT_DETECTION_MODEL_ID = "PaddlePaddle/PP-OCRv5_mobile_det_safetensors"
DEFAULT_RECOGNITION_MODEL_ID = "PaddlePaddle/PP-OCRv5_mobile_rec_safetensors"


_STRONG_PATTERNS = (
    ("sample", re.compile(r"\bSAMPLE\b")),
    ("preview", re.compile(r"\bPREVIEW\b")),
    ("proof", re.compile(r"\bPROOF\b")),
    ("draft", re.compile(r"\bDRAFT\b")),
    ("watermark", re.compile(r"\bWATER[\s_-]*MARK\b")),
    ("copyright", re.compile(r"\bCOPYRIGHT\b")),
    ("confidential", re.compile(r"\bCONFIDENTIAL\b")),
    ("do_not_copy", re.compile(r"\bDO\s+NOT\s+COPY\b")),
    ("ai_generated", re.compile(r"\bAI[\s_-]*GENERATED\b")),
    ("generated_with", re.compile(r"\bGENERATED\s+WITH\b")),
    ("ai_generated_zh", re.compile(r"(?:由\s*)?AI\s*生成", re.IGNORECASE)),
)

_URL_PATTERNS = (
    re.compile(r"\bhttps?://[^\s]+", re.IGNORECASE),
    re.compile(r"\bwww\.[^\s]+", re.IGNORECASE),
    re.compile(
        r"(?<![@\w])(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
        r"(?:com|org|net|io|ai|co|cn|dev|app|me|tv)(?:/[^\s]*)?\b",
        re.IGNORECASE,
    ),
)
_HANDLE_PATTERN = re.compile(r"(?<![\w@])@[a-z0-9_.-]{2,32}\b", re.IGNORECASE)
_DATE_PATTERNS = (
    re.compile(r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b"),
    re.compile(r"(?:19|20)\d{2}年(?:0?[1-9]|1[0-2])月(?:0?[1-9]|[12]\d|3[01])日"),
)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _python_value(value: Any) -> Any:
    """Recursively convert tensor/array/scalar values to JSON-safe values."""
    if isinstance(value, torch.Tensor):
        return _python_value(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _python_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_python_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _python_value(value.tolist())
    if hasattr(value, "item"):
        return _python_value(value.item())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        converted = _python_value(value)
        if isinstance(converted, list):
            converted = converted[0]
        number = float(converted)
    except (TypeError, ValueError, IndexError):
        return default
    return number if math.isfinite(number) else default


def _move_to_device(inputs: Any, device: torch.device) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def _model_inputs(inputs: Any) -> dict[str, Any]:
    if isinstance(inputs, Mapping):
        return {key: value for key, value in inputs.items() if key != "target_sizes"}
    return dict(inputs)


def _polygon_from_box(box: Any, width: int, height: int) -> list[list[float]] | None:
    raw = _python_value(box)
    points: list[list[float]]
    if (
        isinstance(raw, list)
        and len(raw) >= 4
        and all(isinstance(point, list) and len(point) >= 2 for point in raw[:4])
    ):
        points = [[_as_float(point[0]), _as_float(point[1])] for point in raw[:4]]
    elif isinstance(raw, list) and len(raw) >= 4:
        x1, y1, x2, y2 = (_as_float(value) for value in raw[:4])
        points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        return None

    max_x = max(0.0, float(width))
    max_y = max(0.0, float(height))
    clamped = [
        [min(max(point[0], 0.0), max_x), min(max(point[1], 0.0), max_y)]
        for point in points
    ]
    xs = [point[0] for point in clamped]
    ys = [point[1] for point in clamped]
    if max(xs) <= min(xs) or max(ys) <= min(ys):
        return None
    return clamped


def _bounding_box(polygon: Sequence[Sequence[float]]) -> list[float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


class TextWatermarkDetector:
    """PP-OCRv5 proposal expert with conservative watermark arbitration."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        detection_model_id: str = DEFAULT_DETECTION_MODEL_ID,
        recognition_model_id: str = DEFAULT_RECOGNITION_MODEL_ID,
        detection_score_threshold: float = 0.5,
        recognition_score_threshold: float = 0.5,
        edge_ratio: float = 0.08,
        max_regions: int = 4,
        max_total_area_ratio: float = 0.12,
        detection_processor_factory: Callable[[str], Any] | None = None,
        detection_model_factory: Callable[[str], Any] | None = None,
        recognition_processor_factory: Callable[[str], Any] | None = None,
        recognition_model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else _default_device()
        self.detection_model_id = detection_model_id
        self.recognition_model_id = recognition_model_id
        self.detection_score_threshold = float(detection_score_threshold)
        self.recognition_score_threshold = float(recognition_score_threshold)
        self.edge_ratio = min(max(float(edge_ratio), 0.0), 0.5)
        self.max_regions = max(1, int(max_regions))
        self.max_total_area_ratio = min(max(float(max_total_area_ratio), 0.0), 1.0)

        self._detection_processor_factory = detection_processor_factory
        self._detection_model_factory = detection_model_factory
        self._recognition_processor_factory = recognition_processor_factory
        self._recognition_model_factory = recognition_model_factory
        self._detection_processor = None
        self._detection_model = None
        self._recognition_processor = None
        self._recognition_model = None
        self.last_filter_report: dict[str, Any] = self._filter_report()

    @property
    def detection_processor(self) -> Any:
        return self._detection_processor

    @property
    def detection_model(self) -> Any:
        return self._detection_model

    @property
    def recognition_processor(self) -> Any:
        return self._recognition_processor

    @property
    def recognition_model(self) -> Any:
        return self._recognition_model

    @staticmethod
    def _prepare_model(model: Any, device: torch.device) -> Any:
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            evaluated = model.eval()
            if evaluated is not None:
                model = evaluated
        return model

    def _ensure_detection_models(self) -> None:
        if self._detection_processor is not None and self._detection_model is not None:
            return
        if self._detection_processor_factory is None:
            from transformers import AutoImageProcessor

            processor_factory = AutoImageProcessor.from_pretrained
        else:
            processor_factory = self._detection_processor_factory
        if self._detection_model_factory is None:
            from transformers import AutoModelForObjectDetection

            model_factory = AutoModelForObjectDetection.from_pretrained
        else:
            model_factory = self._detection_model_factory

        self._detection_processor = processor_factory(self.detection_model_id)
        self._detection_model = self._prepare_model(
            model_factory(self.detection_model_id), self.device
        )

    def _ensure_recognition_models(self) -> None:
        if self._recognition_processor is not None and self._recognition_model is not None:
            return
        if self._recognition_processor_factory is None:
            from transformers import AutoImageProcessor

            processor_factory = AutoImageProcessor.from_pretrained
        else:
            processor_factory = self._recognition_processor_factory
        if self._recognition_model_factory is None:
            from transformers import AutoModelForTextRecognition

            model_factory = AutoModelForTextRecognition.from_pretrained
        else:
            model_factory = self._recognition_model_factory

        self._recognition_processor = processor_factory(self.recognition_model_id)
        self._recognition_model = self._prepare_model(
            model_factory(self.recognition_model_id), self.device
        )

    def detect(self, image: Image.Image) -> list[dict[str, Any]]:
        """Return JSON-safe OCR proposals with polygons and recognized text."""
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")
        image = image.convert("RGB")
        self._ensure_detection_models()

        detection_inputs = self._detection_processor(
            images=image, return_tensors="pt"
        )
        detection_inputs = _move_to_device(detection_inputs, self.device)
        target_sizes = detection_inputs["target_sizes"]
        with torch.inference_mode():
            detection_outputs = self._detection_model(
                **_model_inputs(detection_inputs)
            )
        processed = self._detection_processor.post_process_object_detection(
            detection_outputs, target_sizes=target_sizes
        )
        if not processed:
            return []

        result = processed[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        proposals: list[dict[str, Any]] = []
        for raw_box, raw_score in zip(boxes, scores):
            score = _as_float(raw_score)
            if score < self.detection_score_threshold:
                continue
            polygon = _polygon_from_box(raw_box, image.width, image.height)
            if polygon is None:
                continue
            box = _bounding_box(polygon)
            crop = image.crop(
                (
                    max(0, math.floor(box[0])),
                    max(0, math.floor(box[1])),
                    min(image.width, math.ceil(box[2])),
                    min(image.height, math.ceil(box[3])),
                )
            )
            text, recognition_score = self._recognize(crop)
            proposals.append(
                {
                    "box": [float(value) for value in box],
                    "polygon": [[float(value) for value in point] for point in polygon],
                    "detection_score": float(score),
                    "text": str(text),
                    "recognition_score": float(recognition_score),
                }
            )
        return proposals

    def _recognize(self, crop: Image.Image) -> tuple[str, float]:
        self._ensure_recognition_models()
        recognition_inputs = self._recognition_processor(
            images=crop, return_tensors="pt"
        )
        recognition_inputs = _move_to_device(recognition_inputs, self.device)
        with torch.inference_mode():
            recognition_outputs = self._recognition_model(
                **_model_inputs(recognition_inputs)
            )
        processed = self._recognition_processor.post_process_text_recognition(
            recognition_outputs
        )
        if not processed:
            return "", 0.0
        first = processed[0]
        if isinstance(first, Mapping):
            return str(first.get("text", "")), _as_float(first.get("score", 0.0))
        return str(first), 0.0

    def filter_watermarks(
        self, proposals: Sequence[Mapping[str, Any]], width: int, height: int
    ) -> list[dict[str, Any]]:
        """Accept strong watermark phrases or edge-bound weak attribution text.

        This is fail-closed: if the complete accepted OCR set exceeds the count
        or area safety budget, the entire set is rejected.  Returning only a
        subset could make a tiled watermark look successfully cleaned.
        """
        image_width = max(0, int(width))
        image_height = max(0, int(height))
        image_area = float(max(1, image_width * image_height))
        candidates: list[tuple[tuple[Any, ...], dict[str, Any], float]] = []

        for index, proposal in enumerate(proposals):
            normalized = self._normalize_filter_candidate(
                proposal, image_width, image_height
            )
            if normalized is None:
                continue
            item, evidence, evidence_rank = normalized
            area = self._box_area(item["box"])
            sort_key = (
                evidence_rank,
                -float(item["recognition_score"]),
                -float(item["detection_score"]),
                float(item["box"][1]),
                float(item["box"][0]),
                float(item["box"][3]),
                float(item["box"][2]),
                unicodedata.normalize("NFKC", item["text"]).casefold(),
                index,
            )
            item["source"] = "generic_ocr"
            item["evidence"] = evidence
            item["detection_tier"] = "generic_ocr"
            candidates.append((sort_key, item, area))

        total_area = sum(candidate[2] for candidate in candidates)
        total_area_ratio = total_area / image_area
        overflow_reasons = []
        if len(candidates) > self.max_regions:
            overflow_reasons.append("max_regions")
        if total_area_ratio > self.max_total_area_ratio + 1e-12:
            overflow_reasons.append("max_total_area_ratio")
        if overflow_reasons:
            self.last_filter_report = self._filter_report(
                overflow=True,
                reasons=overflow_reasons,
                candidate_count=len(candidates),
                accepted_count=0,
                total_area_ratio=total_area_ratio,
            )
            return []

        candidates.sort(key=lambda candidate: candidate[0])
        accepted = [candidate[1] for candidate in candidates]
        self.last_filter_report = self._filter_report(
            candidate_count=len(candidates),
            accepted_count=len(accepted),
            total_area_ratio=total_area_ratio,
        )
        return accepted

    def _normalize_filter_candidate(
        self, proposal: Mapping[str, Any], width: int, height: int
    ) -> tuple[dict[str, Any], list[str], int] | None:
        text = str(proposal.get("text", "")).strip()
        if not text:
            return None
        detection_score = _as_float(proposal.get("detection_score", 0.0))
        recognition_score = _as_float(proposal.get("recognition_score", 0.0))
        if detection_score < self.detection_score_threshold:
            return None
        if recognition_score < self.recognition_score_threshold:
            return None

        polygon = _polygon_from_box(
            proposal.get("polygon", proposal.get("box")), width, height
        )
        if polygon is None:
            return None
        box = _bounding_box(polygon)
        normalized_text = unicodedata.normalize("NFKC", text)
        uppercase_text = re.sub(r"\s+", " ", normalized_text).strip().upper()
        strong = [
            f"strong_lexical:{name}"
            for name, pattern in _STRONG_PATTERNS
            if pattern.search(uppercase_text)
        ]
        if strong:
            evidence = strong
            evidence_rank = 0
        else:
            weak = self._weak_evidence(normalized_text)
            if not weak or not self._is_edge_box(box, width, height):
                return None
            evidence = ["edge_geometry", *weak]
            evidence_rank = 1

        item = {
            "box": [float(value) for value in box],
            "polygon": [[float(value) for value in point] for point in polygon],
            "detection_score": float(detection_score),
            "text": text,
            "recognition_score": float(recognition_score),
        }
        return item, evidence, evidence_rank

    @staticmethod
    def _weak_evidence(text: str) -> list[str]:
        evidence = []
        if any(pattern.search(text) for pattern in _URL_PATTERNS):
            evidence.append("weak_lexical:url")
        if _HANDLE_PATTERN.search(text):
            evidence.append("weak_lexical:handle")
        if any(pattern.search(text) for pattern in _DATE_PATTERNS):
            evidence.append("weak_lexical:date")
        return evidence

    def _is_edge_box(self, box: Sequence[float], width: int, height: int) -> bool:
        if width <= 0 or height <= 0:
            return False
        x1, y1, x2, y2 = (float(value) for value in box)
        x_margin = width * self.edge_ratio
        y_margin = height * self.edge_ratio
        return (
            x1 <= x_margin
            or y1 <= y_margin
            or x2 >= width - x_margin
            or y2 >= height - y_margin
        )

    @staticmethod
    def _box_area(box: Sequence[float]) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(
            0.0, float(box[3]) - float(box[1])
        )

    def _filter_report(
        self,
        *,
        overflow: bool = False,
        reasons: Sequence[str] = (),
        candidate_count: int = 0,
        accepted_count: int = 0,
        total_area_ratio: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "overflow": bool(overflow),
            "reasons": [str(reason) for reason in reasons],
            "candidate_count": int(candidate_count),
            "accepted_count": int(accepted_count),
            "total_area_ratio": float(total_area_ratio),
            "max_regions": int(self.max_regions),
            "max_total_area_ratio": float(self.max_total_area_ratio),
        }


__all__ = [
    "DEFAULT_DETECTION_MODEL_ID",
    "DEFAULT_RECOGNITION_MODEL_ID",
    "TextWatermarkDetector",
]
