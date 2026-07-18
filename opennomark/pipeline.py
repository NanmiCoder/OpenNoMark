"""Core pipeline: detect watermarks and remove them."""

import os
from PIL import Image, ImageChops, ImageFilter

from .inpainter import LamaInpainter
from .localizer import WatermarkLocalizer


class WatermarkRemovalPipeline:
    def __init__(self, device=None, verbose=True):
        self.verbose = verbose
        if self.verbose:
            print("Loading inpainting model...")
        self.device = device
        self.localizer = WatermarkLocalizer(device=device)
        self.inpainter = LamaInpainter(device=device)
        if self.verbose:
            print("Inpainting model loaded.")

    def process(self, image_path, output_path=None, save_debug=False):
        """Process a single image. Returns (result_image, metadata).

        Pipeline:
          1. Ask the unified localizer for evidence-backed regions.  It may use
             a precise shape expert or open-vocabulary proposals, but never a
             filename/provider branch.
          2. Inpaint every accepted region on a local crop with its own mask.
          3. Return one metadata contract for every visual watermark family.
        """
        image = Image.open(image_path).convert("RGB")
        filename = os.path.basename(image_path)
        regions, localization = self.localizer.localize(image)
        public_regions = [region.as_metadata() for region in regions]
        methods_used = []

        metadata = {
            "input": image_path,
            "methods": methods_used,
            "localization": localization,
            "total_detections": localization["total_proposals"],
            "watermarks_found": len(regions),
            "regions": public_regions,
            # Backwards-compatible alias for clients written before the unified
            # region contract.  New code should consume ``regions``.
            "boxes": public_regions,
        }
        automatic_removal_blocked = bool(
            localization.get("safety", {}).get("automatic_removal_blocked")
        )

        if not regions:
            if automatic_removal_blocked:
                metadata["validation"] = {
                    "passed": False,
                    "attempts": 0,
                    "overlapping_residual_regions": [],
                    "reason": "candidate_budget_exceeded",
                }
                metadata["status"] = "partial"
            else:
                metadata["status"] = "no_watermark"
            return image, metadata

        result = image
        for region in regions:
            result = self.inpainter.inpaint_local(result, region.mask)
            method = f"{region.source}_local_lama"
            if method not in methods_used:
                methods_used.append(method)

        residual_regions = self.localizer.localize_residuals(result, regions)
        overlapping = self._overlapping_regions(regions, residual_regions)
        attempts = 1
        if overlapping:
            # A detector box can hug the opaque glyph core and miss a faint
            # antialiasing fringe. Retry once with a slightly grown version of
            # the same evidence-backed mask; never expand to unrelated regions.
            for region in regions:
                if any(self._overlap(region.box, residual.box) >= 0.25 for residual in overlapping):
                    expanded = region.mask.filter(ImageFilter.MaxFilter(9))
                    result = self.inpainter.inpaint_local(result, expanded)
            attempts = 2
            residual_regions = self.localizer.localize_residuals(result, regions)
            overlapping = self._overlapping_regions(regions, residual_regions)

        metadata["methods"] = methods_used
        metadata["validation"] = {
            "passed": not overlapping and not automatic_removal_blocked,
            "attempts": attempts,
            "overlapping_residual_regions": [region.as_metadata() for region in overlapping],
        }
        if automatic_removal_blocked:
            metadata["validation"]["reason"] = "candidate_budget_exceeded"
        metadata["status"] = (
            "cleaned"
            if not overlapping and not automatic_removal_blocked
            else "partial"
        )

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
                combined_mask = Image.new("L", image.size, 0)
                for region in regions:
                    x1, y1, x2, y2 = region.box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    combined_mask = ImageChops.lighter(combined_mask, region.mask)
                debug_img.save(os.path.join(debug_dir, f"debug_{filename}"))
                combined_mask.save(os.path.join(debug_dir, f"mask_{filename}"))

        return result, metadata

    @classmethod
    def _overlapping_regions(cls, original_regions, residual_regions):
        return [
            residual
            for residual in residual_regions
            if any(cls._overlap(original.box, residual.box) >= 0.25 for original in original_regions)
        ]

    @staticmethod
    def _overlap(reference, other):
        x1 = max(float(reference[0]), float(other[0]))
        y1 = max(float(reference[1]), float(other[1]))
        x2 = min(float(reference[2]), float(other[2]))
        y2 = min(float(reference[3]), float(other[3]))
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        reference_area = max(0.0, float(reference[2]) - float(reference[0])) * max(
            0.0, float(reference[3]) - float(reference[1])
        )
        return intersection / reference_area if reference_area else 0.0

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
