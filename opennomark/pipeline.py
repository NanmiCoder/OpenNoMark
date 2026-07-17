"""Core pipeline: detect watermarks and remove them."""

import os
from PIL import Image

from .detector import WatermarkDetector
from .inpainter import LamaInpainter
from .gemini_alpha import create_gemini_mask, detect_gemini_watermark


class WatermarkRemovalPipeline:
    def __init__(self, device=None, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("Loading inpainting model...")
        self.device = device
        self.detector = None
        self.inpainter = LamaInpainter(device=device)
        if self.verbose:
            print("Inpainting model loaded.")

    def _get_detector(self):
        """Load the generic 600M OWLv2 model only when it is actually needed."""
        if self.detector is None:
            self.detector = WatermarkDetector(device=self.device)
        return self.detector

    def process(self, image_path, output_path=None, save_debug=False):
        """Process a single image. Returns (result_image, metadata).

        Pipeline:
          1. Score Gemini's catalog anchors with the trained spatial+edge
             detector (including the current 96px/192px-margin layout).
          2. For Gemini, run LaMa on a tight sparkle-shaped local crop. This is
             deterministic, fast, and does not depend on OWLv2 recognizing a
             tiny low-contrast icon.
          3. Only when no Gemini mark is found, use generic OWLv2+LaMa. This
             avoids deleting unrelated UI icons from Gemini screenshots.
        """
        image = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)
        methods_used = []
        working = image
        mask = None

        # Dedicated Gemini path: catalog localization + shape-aware local LaMa.
        gemini_det = detect_gemini_watermark(working, source_hint=filename)
        gemini_found = gemini_det.get("found", False)
        if gemini_found:
            mask = create_gemini_mask(working.size, gemini_det)
            working = self.inpainter.inpaint_local(working, mask)
            methods_used.append("gemini_catalog_lama")
            all_boxes = []
            filtered = []
        else:
            # Generic path is intentionally isolated from confirmed Gemini
            # images: low-threshold OWLv2 otherwise erases unrelated UI icons.
            detector = self._get_detector()
            all_boxes = detector.detect(working)
            filtered = detector.filter_watermarks(all_boxes, working.width, working.height)

        metadata = {
            "input": image_path,
            "methods": methods_used,
            "total_detections": len(all_boxes),
            "watermarks_found": len(filtered) + (1 if gemini_found else 0),
            "boxes": filtered,
        }

        if gemini_found:
            metadata["gemini_detection"] = {
                "layout": gemini_det["layout"],
                "position": (gemini_det["x"], gemini_det["y"]),
                "logo_size": gemini_det["logo_size"],
                "margin": gemini_det["margin"],
                "spatial_score": gemini_det["spatial_score"],
                "gradient_score": gemini_det["gradient_score"],
                "confidence": gemini_det["confidence"],
                "decision": gemini_det["decision"],
            }

        if not filtered and not methods_used:
            metadata["status"] = "no_watermark"
            return image, metadata

        if gemini_found:
            result = working
        elif filtered:
            mask = self.inpainter.create_mask((working.width, working.height), filtered)
            result = self.inpainter.inpaint(working, mask)
            methods_used.append("owlv2_lama")
        else:
            result = working

        metadata["methods"] = methods_used
        metadata["status"] = "cleaned"

        # Save outputs
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            # Preserve original format
            ext = os.path.splitext(image_path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                result.save(output_path, quality=95)
            else:
                result.save(output_path)

            if save_debug:
                debug_dir = os.path.dirname(output_path)
                from PIL import ImageDraw
                debug_img = image.copy()
                draw = ImageDraw.Draw(debug_img)
                if gemini_found:
                    gx, gy = gemini_det["x"], gemini_det["y"]
                    gs = gemini_det["logo_size"]
                    draw.rectangle([gx, gy, gx + gs, gy + gs], outline="cyan", width=3)
                for item in filtered:
                    x1, y1, x2, y2 = item["box"]
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                debug_img.save(os.path.join(debug_dir, f"debug_{filename}"))
                if mask is not None:
                    mask.save(os.path.join(debug_dir, f"mask_{filename}"))

        return result, metadata

    def process_batch(self, image_paths, output_dir, save_debug=False, callback=None):
        """Process multiple images. callback(i, total, metadata) for progress."""
        os.makedirs(output_dir, exist_ok=True)
        results = []

        for i, path in enumerate(image_paths):
            filename = os.path.basename(path)
            # Normalize output filename
            name, ext = os.path.splitext(filename)
            if not ext or ext.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                ext = ".png"
            out_name = f"clean_{name}{ext}"
            out_path = os.path.join(output_dir, out_name)

            _, meta = self.process(path, out_path, save_debug=save_debug)
            meta["output"] = out_path
            results.append(meta)

            if callback:
                callback(i + 1, len(image_paths), meta)

        return results
