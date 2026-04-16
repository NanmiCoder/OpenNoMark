"""Gemini watermark removal via reverse alpha blending.

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

import os
import numpy as np
from PIL import Image

# Pre-computed alpha maps for Gemini's sparkle watermark
_ALPHA_DIR = os.path.join(os.path.dirname(__file__), "assets")
_ALPHA_48 = None
_ALPHA_96 = None

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


def detect_gemini_watermark(image, min_confidence=0.90):
    """Detect Gemini sparkle watermark position and size.

    STRICT default threshold (0.90): we only invoke the lossless alpha
    path when the template match is near-perfect, because any alpha-map
    misalignment or Gemini-version mismatch leaves visible 'dent' artefacts
    that are WORSE than letting LaMa handle the region.

    Returns dict with: found, x, y, logo_size, alpha_map, confidence
    (or found=False).
    """
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    candidates = []
    if w > 1024 and h > 1024:
        candidates.append((96, 64))
        candidates.append((48, 32))
    else:
        candidates.append((48, 32))
        candidates.append((96, 64))

    gray = np.mean(img_np[:, :, :3], axis=2).astype(np.float64)

    best = None
    for logo_size, margin in candidates:
        alpha_map = _load_alpha(logo_size)
        template = _make_template_gray(alpha_map)

        x = w - margin - logo_size
        y = h - margin - logo_size
        if x < 0 or y < 0:
            continue

        for dx in range(-16, 17, 2):
            for dy in range(-16, 17, 2):
                cx, cy = x + dx, y + dy
                if cx < 0 or cy < 0 or cx + logo_size > w or cy + logo_size > h:
                    continue
                patch = gray[cy:cy + logo_size, cx:cx + logo_size]
                score = _ncc(patch, template.astype(np.float64))
                if best is None or score > best["confidence"]:
                    best = {
                        "found": True,
                        "x": cx,
                        "y": cy,
                        "logo_size": logo_size,
                        "alpha_map": alpha_map,
                        "confidence": score,
                    }

    if best is None or best["confidence"] < min_confidence:
        return {"found": False, "confidence": best["confidence"] if best else 0.0}

    # Fine search around best position
    bx, by = best["x"], best["y"]
    logo_size = best["logo_size"]
    alpha_map = best["alpha_map"]
    template = _make_template_gray(alpha_map).astype(np.float64)

    for dx in range(-3, 4):
        for dy in range(-3, 4):
            cx, cy = bx + dx, by + dy
            if cx < 0 or cy < 0 or cx + logo_size > w or cy + logo_size > h:
                continue
            patch = gray[cy:cy + logo_size, cx:cx + logo_size]
            score = _ncc(patch, template)
            if score > best["confidence"]:
                best["x"] = cx
                best["y"] = cy
                best["confidence"] = score

    return best


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
