"""Gemini watermark detection, masking, and reverse-alpha utilities.

Mathematically reverses the alpha compositing formula used by Google Gemini
to embed its sparkle watermark. Uses LINEAR-LIGHT (sRGB-decoded) math so
the recovered pixels match the original — eliminates the ``darkening dent''
that appeared when reverse-alpha was done directly on sRGB bytes.

Decision flow:
  1. detect a candidate at the standard position (cheap NCC template match,
     low threshold 0.30 — just needs a signal to exist).
  2. reverse-alpha in linear-light space.
  3. post-hoc quality check: was the sparkle actually eliminated, and is
     the region genuinely brighter than the surrounding background? Only
     accept the result if both pass. This filters out random correlations
     on non-Gemini images.

Based on research from GargantuaX/gemini-watermark-remover with
linear-light correction and posterior-quality gating added by us.
"""

import json
import os
import re

import cv2
import numpy as np
from PIL import Image

# Pre-computed alpha maps for Gemini's sparkle watermark
_ALPHA_DIR = os.path.join(os.path.dirname(__file__), "assets")
_ALPHA_48 = None
_ALPHA_96 = None

_DETECTOR_MODEL = None
_DETECTOR_MODEL_PATH = os.path.join(_ALPHA_DIR, "gemini_detector.json")

# GargantuaX/gemini-watermark-remover's current catalog contains three
# important visible-watermark anchors.  The 96/192 layout is the May 2026
# variant used by current 2K Gemini outputs.  It must be tried before the old
# 96/64 layout: the two positions are 128 pixels apart.
_KNOWN_LAYOUTS = (
    {"name": "gemini_2k_20260520", "logo_size": 96, "margin": 192},
    {"name": "gemini_1k_current", "logo_size": 48, "margin": 32},
    {"name": "gemini_large_legacy", "logo_size": 96, "margin": 64},
)

_DEFAULT_DETECTOR_MODEL = {
    "version": 1,
    "decision": {
        "min_spatial_score": 0.18,
        "min_gradient_score": 0.18,
        "hinted_min_spatial_score": 0.15,
        "hinted_min_gradient_score": 0.08,
    },
}

LOGO_VALUE = 255  # White watermark (sRGB)
LOGO_LINEAR = 1.0  # White in linear-light space
ALPHA_THRESHOLD = 0.002
MAX_ALPHA = 0.99


# ---------------- sRGB <-> linear-light (IEC 61966-2-1) ---------------- #

def _srgb_to_linear(x):
    """x in [0, 1] sRGB -> linear light."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _linear_to_srgb(x):
    """x in [0, 1] linear light -> sRGB."""
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)


def _load_alpha(size):
    """Load pre-computed alpha map."""
    global _ALPHA_48, _ALPHA_96
    if size == 48:
        if _ALPHA_48 is None:
            _ALPHA_48 = np.load(os.path.join(_ALPHA_DIR, "gemini_alpha_48.npy"))
        return _ALPHA_48
    else:
        if _ALPHA_96 is None:
            _ALPHA_96 = np.load(os.path.join(_ALPHA_DIR, "gemini_alpha_96.npy"))
        return _ALPHA_96


def _ncc(patch, template):
    """Normalized Cross-Correlation between a grayscale patch and template."""
    p = patch.astype(np.float64).ravel()
    t = template.astype(np.float64).ravel()
    p = p - p.mean()
    t = t - t.mean()
    denom = np.sqrt(np.sum(p ** 2) * np.sum(t ** 2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(p * t) / denom)


def _make_template_gray(alpha_map):
    """Create a grayscale template from the alpha map (white logo on black)."""
    return (alpha_map * 255).astype(np.uint8)


def _load_detector_model():
    """Load thresholds calibrated by ``scripts/train_gemini_detector.py``."""
    global _DETECTOR_MODEL
    if _DETECTOR_MODEL is not None:
        return _DETECTOR_MODEL

    model = _DEFAULT_DETECTOR_MODEL
    try:
        with open(_DETECTOR_MODEL_PATH, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict) and isinstance(loaded.get("decision"), dict):
            model = loaded
    except (OSError, ValueError):
        pass

    _DETECTOR_MODEL = model
    return model


def _sobel_magnitude(values):
    values = values.astype(np.float32, copy=False)
    gx = cv2.Sobel(values, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(values, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _score_patch(gray_patch, alpha_map):
    """Return spatial and edge-template correlations for one candidate."""
    spatial = _ncc(gray_patch, alpha_map)
    gradient = _ncc(_sobel_magnitude(gray_patch), _sobel_magnitude(alpha_map))
    return float(spatial), float(gradient)


def _is_gemini_source_hint(source_hint):
    if not source_hint:
        return False
    name = os.path.basename(os.fspath(source_hint))
    return re.match(r"(?i)^gemini[_ -]generated[_ -]image", name) is not None


def _candidate_at(gray, layout, x, y):
    logo_size = layout["logo_size"]
    h, w = gray.shape
    if x < 0 or y < 0 or x + logo_size > w or y + logo_size > h:
        return None
    alpha_map = _load_alpha(logo_size)
    patch = gray[y:y + logo_size, x:x + logo_size]
    spatial, gradient = _score_patch(patch, alpha_map)
    return {
        "x": x,
        "y": y,
        "logo_size": logo_size,
        "margin": layout["margin"],
        "layout": layout["name"],
        "alpha_map": alpha_map,
        "spatial_score": spatial,
        "gradient_score": gradient,
        "ranking_score": spatial + 0.8 * max(0.0, gradient),
    }


def _candidate_layouts(width, height):
    """Return only layouts plausible for the Gemini output resolution tier.

    Current 2K/4K outputs have a long side of at least 2048 and a short side
    of at least 600. Narrower or smaller preview/1K outputs use the 48px mark.
    Keeping tiers separate prevents an unrelated tiny corner icon from beating
    the real 96px watermark on a large image.
    """
    long_side = max(width, height)
    short_side = min(width, height)
    if long_side >= 2048 and short_side >= 600:
        return (_KNOWN_LAYOUTS[0], _KNOWN_LAYOUTS[2])
    return (_KNOWN_LAYOUTS[1], _KNOWN_LAYOUTS[2])


def _best_layout_candidate(gray, layout):
    """Search locally around a catalog anchor, coarse then pixel-precise."""
    h, w = gray.shape
    logo_size = layout["logo_size"]
    margin = layout["margin"]
    base_x = w - margin - logo_size
    base_y = h - margin - logo_size
    best = None

    # Position drift exists in older Gemini exports.  The search remains local
    # so arbitrary star-shaped content elsewhere in the image is not promoted.
    for dx in range(-24, 25, 4):
        for dy in range(-24, 25, 4):
            candidate = _candidate_at(gray, layout, base_x + dx, base_y + dy)
            if candidate is not None and (
                best is None or candidate["ranking_score"] > best["ranking_score"]
            ):
                best = candidate

    if best is None:
        return None

    coarse_x, coarse_y = best["x"], best["y"]
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            candidate = _candidate_at(gray, layout, coarse_x + dx, coarse_y + dy)
            if candidate is not None and candidate["ranking_score"] > best["ranking_score"]:
                best = candidate
    return best


def detect_gemini_watermark(image, min_confidence=None, source_hint=None):
    """Detect a Gemini sparkle with a trained catalog-template detector.

    Gemini uses a small catalog of output dimensions and anchors.  We score
    those anchors with both luminance and Sobel-edge correlation, then apply
    thresholds calibrated on real Gemini positives and hard negatives.  This
    replaces the previous fixed 96/64-only search and its brittle 0.90 NCC
    gate. ``source_hint`` is only a weak fallback for borderline evidence.
    """
    img_np = np.array(image.convert("RGB"))
    rgb = img_np[:, :, :3].astype(np.float32) / 255.0
    gray = (
        0.2126 * rgb[:, :, 0] +
        0.7152 * rgb[:, :, 1] +
        0.0722 * rgb[:, :, 2]
    )

    candidates = [
        candidate
        for layout in _candidate_layouts(gray.shape[1], gray.shape[0])
        if (candidate := _best_layout_candidate(gray, layout)) is not None
    ]
    if not candidates:
        return {"found": False, "confidence": 0.0}

    best = max(candidates, key=lambda item: item["ranking_score"])
    model = _load_detector_model()
    decision = model.get("decision", {})
    min_spatial = float(decision.get("min_spatial_score", 0.18))
    min_gradient = float(decision.get("min_gradient_score", 0.18))
    hinted_min_spatial = float(decision.get("hinted_min_spatial_score", 0.15))
    hinted_min_gradient = float(decision.get("hinted_min_gradient_score", 0.08))

    spatial = best["spatial_score"]
    gradient = best["gradient_score"]
    source_hinted = _is_gemini_source_hint(source_hint)
    trained_match = spatial >= min_spatial and gradient >= min_gradient
    hinted_match = (
        source_hinted and
        spatial >= hinted_min_spatial and
        gradient >= hinted_min_gradient
    )
    confidence = float(np.clip((max(0.0, spatial) + max(0.0, gradient)) / 2.0, 0.0, 1.0))
    found = trained_match or hinted_match
    if min_confidence is not None:
        # Preserve the old experiment/debug API: an explicit confidence value
        # replaces the trained decision (notably ``-1`` returns the best raw
        # catalog candidate). Production callers leave this as ``None``.
        found = confidence >= min_confidence

    best.update({
        "found": found,
        "confidence": confidence,
        "source_hinted": source_hinted,
        "decision": "trained_match" if trained_match else ("source_hint" if hinted_match else "rejected"),
        "model_version": model.get("version", 1),
    })
    return best


def create_gemini_mask(image_size, detection, alpha_threshold=0.02, dilation=5, feather=2):
    """Create a tight sparkle-shaped mask for deterministic local inpainting."""
    width, height = image_size
    x, y = int(detection["x"]), int(detection["y"])
    logo_size = int(detection["logo_size"])
    alpha_map = detection["alpha_map"]
    if alpha_map.shape != (logo_size, logo_size):
        alpha_map = cv2.resize(alpha_map, (logo_size, logo_size), interpolation=cv2.INTER_LINEAR)

    core = (alpha_map > alpha_threshold).astype(np.uint8) * 255
    if dilation > 1:
        kernel_size = max(1, int(dilation))
        core = cv2.dilate(core, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
    if feather > 0:
        core = cv2.GaussianBlur(core, (0, 0), sigmaX=float(feather))

    mask = np.zeros((height, width), dtype=np.uint8)
    x2, y2 = min(width, x + logo_size), min(height, y + logo_size)
    if x < x2 and y < y2:
        mask[y:y2, x:x2] = core[:y2 - y, :x2 - x]
    return Image.fromarray(mask, mode="L")


def _find_best_gain(img_patch, alpha_map):
    """Find optimal alpha gain in LINEAR-LIGHT space (handles JPEG drift)."""
    gray_lin = _srgb_to_linear(img_patch.astype(np.float64) / 255.0).mean(axis=2)

    best_gain = 1.0
    best_residual = float("inf")

    # Coarse search — wider now that we're in linear space where optimal
    # gain is typically 0.5–1.5 for 8-bit sRGB sources.
    for gain in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0]:
        alpha = np.minimum(alpha_map * gain, MAX_ALPHA)
        restored = np.where(alpha > ALPHA_THRESHOLD,
                            (gray_lin - alpha * LOGO_LINEAR) / (1.0 - alpha),
                            gray_lin)
        restored = np.clip(restored, 0.0, 1.0)
        residual_ncc = abs(_ncc(restored, alpha_map))
        if residual_ncc < best_residual:
            best_residual = residual_ncc
            best_gain = gain

    for gain in np.arange(max(0.05, best_gain - 0.1), best_gain + 0.11, 0.02):
        alpha = np.minimum(alpha_map * gain, MAX_ALPHA)
        restored = np.where(alpha > ALPHA_THRESHOLD,
                            (gray_lin - alpha * LOGO_LINEAR) / (1.0 - alpha),
                            gray_lin)
        restored = np.clip(restored, 0.0, 1.0)
        residual_ncc = abs(_ncc(restored, alpha_map))
        if residual_ncc < best_residual:
            best_residual = residual_ncc
            best_gain = gain

    return best_gain


def _border_mismatch(img_np_out, detection):
    """Abs diff between mean gray inside alpha (cleaned) and 6px ring around it.

    A low value (< ~3 gray levels) means the reverse-alpha result blends
    seamlessly with the surrounding background. A high value means the
    cleaned region still stands out as a visible patch — ie. a dent.
    """
    x, y = detection["x"], detection["y"]
    ls = detection["logo_size"]
    am = detection["alpha_map"]
    h, w = img_np_out.shape[:2]
    gray = img_np_out.astype(np.float32).mean(axis=2)

    m = am > 0.05
    if m.sum() < 10:
        return 9e9
    inner = gray[y:y + ls, x:x + ls][m].mean()

    rw = 6
    ring_parts = []
    if y - rw >= 0:
        ring_parts.append(gray[y - rw:y, x:x + ls].ravel())
    if y + ls + rw <= h:
        ring_parts.append(gray[y + ls:y + ls + rw, x:x + ls].ravel())
    if x - rw >= 0:
        ring_parts.append(gray[y:y + ls, x - rw:x].ravel())
    if x + ls + rw <= w:
        ring_parts.append(gray[y:y + ls, x + ls:x + ls + rw].ravel())
    if not ring_parts:
        return 9e9
    outer = np.concatenate(ring_parts).mean()
    return float(abs(inner - outer))


def _quality_accept(conf_before, border_mismatch):
    """Posterior-quality decision for the reverse-alpha path.

    ACCEPT only when:
      (a) the template match was near-perfect (conf >= 0.95), AND
      (b) the reversed patch blends seamlessly with its surroundings
          (border mismatch < 3 gray levels).

    Rationale: empirically, anything below these thresholds leaves a
    visible dark/light diamond artifact that looks worse than LaMa's
    texture-aware inpainting.
    """
    return conf_before >= 0.95 and border_mismatch < 3.0


def remove_gemini_watermark(image, detection=None):
    """Remove Gemini watermark via reverse alpha blending in linear light.

    Returns (PIL.Image, meta). If the posterior quality check fails, the
    returned image is IDENTICAL to the input and meta['status'] == 'rejected'
    — the caller should fall back to generic inpainting.
    """
    if detection is None:
        detection = detect_gemini_watermark(image)

    if not detection.get("found", False):
        return image, {"method": "gemini_alpha", "status": "no_watermark"}

    img_np = np.array(image).astype(np.uint8)
    x, y = detection["x"], detection["y"]
    logo_size = detection["logo_size"]
    alpha_map = detection["alpha_map"]

    # Extract patch
    patch_u8 = img_np[y:y + logo_size, x:x + logo_size].copy()

    # Find gain in linear light
    gain = _find_best_gain(patch_u8, alpha_map)

    # Reverse-alpha in linear-light space
    patch_lin = _srgb_to_linear(patch_u8.astype(np.float64) / 255.0)
    alpha = np.minimum(alpha_map * gain, MAX_ALPHA)
    mask = alpha > ALPHA_THRESHOLD
    out_lin = patch_lin.copy()
    for c in range(3):
        ch = patch_lin[:, :, c]
        restored = np.where(mask,
                            (ch - alpha * LOGO_LINEAR) / (1.0 - alpha),
                            ch)
        out_lin[:, :, c] = np.clip(restored, 0.0, 1.0)
    cleaned_patch = (_linear_to_srgb(out_lin) * 255).clip(0, 255).astype(np.uint8)

    # Build provisional result to measure border mismatch
    out = img_np.copy()
    out[y:y + logo_size, x:x + logo_size] = cleaned_patch

    # Posterior quality check: does the cleaned patch blend with surroundings?
    border = _border_mismatch(out, detection)
    accept = _quality_accept(detection["confidence"], border)

    if not accept:
        return image, {
            "method": "gemini_alpha",
            "status": "rejected",
            "confidence": detection["confidence"],
            "border_mismatch": border,
            "gain": gain,
            "position": (x, y),
            "logo_size": logo_size,
        }

    return Image.fromarray(out), {
        "method": "gemini_alpha",
        "status": "cleaned",
        "confidence": detection["confidence"],
        "border_mismatch": border,
        "gain": gain,
        "position": (x, y),
        "logo_size": logo_size,
    }
