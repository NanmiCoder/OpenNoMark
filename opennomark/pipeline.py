"""Core pipeline: detect watermarks and remove them."""

import os
from PIL import Image

from .detector import WatermarkDetector
from .inpainter import LamaInpainter
from .gemini_alpha import detect_gemini_watermark, remove_gemini_watermark


class WatermarkRemovalPipeline:
    def __init__(self, device=None):
        print("Loading models...")
        self.detector = WatermarkDetector(device=device)
        self.inpainter = LamaInpainter(device=device)
        print("Models loaded.")

    def process(self, image_path, output_path=None, save_debug=False):
        """Process a single image. Returns (result_image, metadata).

        Pipeline:
          1. Search for a Gemini sparkle candidate at the standard position
             (low NCC threshold 0.30 — we cast a wide net).
          2. Reverse-alpha in LINEAR-LIGHT space. A posterior quality check
             inside remove_gemini_watermark gates acceptance — false
             positives on non-Gemini images are automatically rejected.
          3. ALWAYS run OWLv2+LaMa on the (possibly alpha-cleaned) image to
             catch any additional watermarks (text overlays, etc).
        """
        image = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)
        methods_used = []
        working = image

        # Stage 1: Gemini sparkle (lossless, linear-light reverse alpha)
        gemini_det = detect_gemini_watermark(working)
        alpha_info = None
        if gemini_det.get("found"):
            candidate, info = remove_gemini_watermark(working, gemini_det)
            if info.get("status") == "cleaned":
                working = candidate
                alpha_info = info
                methods_used.append("gemini_alpha")
            # else: posterior quality rejected -- fall through to OWLv2+LaMa

        # Stage 2: OWLv2 + LaMa for any remaining watermarks
        all_boxes = self.detector.detect(working)
        filtered = self.detector.filter_watermarks(all_boxes, working.width, working.height)

        # If we already alpha-cleaned the Gemini sparkle, filter out boxes at that position
        if alpha_info is not None and alpha_info.get("status") == "cleaned":
            ax, ay = alpha_info["position"]
            asize = alpha_info["logo_size"]
            def _overlaps_alpha_region(box):
                x1, y1, x2, y2 = box["box"]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                return (ax - 10 <= cx <= ax + asize + 10 and
                        ay - 10 <= cy <= ay + asize + 10)
            filtered = [b for b in filtered if not _overlaps_alpha_region(b)]

        metadata = {
            "input": image_path,
            "methods": methods_used,
            "total_detections": len(all_boxes),
            "watermarks_found": len(filtered) + (1 if alpha_info and alpha_info.get("status") == "cleaned" else 0),
            "boxes": filtered,
        }

        if not filtered and not methods_used:
            metadata["status"] = "no_watermark"
            return image, metadata

        if filtered:
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
                for item in filtered:
                    x1, y1, x2, y2 = item["box"]
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                debug_img.save(os.path.join(debug_dir, f"debug_{filename}"))
                if filtered:
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
