"""Unit tests for the full pipeline."""

import os
import pytest
from PIL import Image


class TestPipeline:
    """Test the complete watermark removal pipeline."""

    @pytest.fixture(scope="class")
    def pipeline(self):
        from opennomark.pipeline import WatermarkRemovalPipeline
        return WatermarkRemovalPipeline(device="cpu")

    def test_process_synthetic(self, pipeline, sample_image, output_dir):
        out_path = os.path.join(output_dir, "clean_test.png")
        result_img, meta = pipeline.process(sample_image, out_path)
        assert result_img is not None
        assert result_img.size[0] > 0
        assert meta["status"] in ("cleaned", "no_watermark")
        assert "watermarks_found" in meta

    def test_process_no_output_path(self, pipeline, sample_image):
        result_img, meta = pipeline.process(sample_image)
        assert result_img is not None
        assert meta["status"] in ("cleaned", "no_watermark")

    def test_candidate_budget_overflow_returns_partial_without_edit(self, tmp_path):
        from opennomark.pipeline import WatermarkRemovalPipeline

        class BlockedLocalizer:
            def localize(self, image):
                return [], {
                    "total_proposals": 7,
                    "accepted_regions": 0,
                    "experts": ["open_vocabulary", "ocr_text"],
                    "safety": {
                        "automatic_removal_blocked": True,
                        "overflow": [
                            {"expert": "ocr_text", "reason": "max_regions"}
                        ],
                    },
                }

        source = tmp_path / "tiled.png"
        original = Image.new("RGB", (80, 60), color=(12, 34, 56))
        original.save(source)
        pipeline = WatermarkRemovalPipeline.__new__(WatermarkRemovalPipeline)
        pipeline.verbose = False
        pipeline.device = "cpu"
        pipeline.localizer = BlockedLocalizer()
        pipeline.inpainter = None

        result, metadata = pipeline.process(str(source))

        assert result.tobytes() == original.tobytes()
        assert metadata["status"] == "partial"
        assert metadata["watermarks_found"] == 0
        assert metadata["validation"] == {
            "passed": False,
            "attempts": 0,
            "overlapping_residual_regions": [],
            "reason": "candidate_budget_exceeded",
        }

    def test_process_batch(self, pipeline, sample_images_dir, output_dir):
        import glob
        paths = sorted(
            glob.glob(os.path.join(sample_images_dir, "*.png"))
            + glob.glob(os.path.join(sample_images_dir, "*.jpg"))
            + glob.glob(os.path.join(sample_images_dir, "*.jpeg"))
        )
        assert len(paths) == 3

        progress_calls = []
        def on_progress(i, total, meta):
            progress_calls.append((i, total, meta["status"]))

        results = pipeline.process_batch(paths, output_dir, callback=on_progress)
        assert len(results) == 3
        assert len(progress_calls) == 3
        for r in results:
            assert r["status"] in ("cleaned", "no_watermark")

    def test_process_batch_with_debug(self, pipeline, sample_images_dir, tmp_path):
        import glob
        out = str(tmp_path / "debug_out")
        paths = glob.glob(os.path.join(sample_images_dir, "*.png"))
        results = pipeline.process_batch(paths, out, save_debug=True)
        # Check debug files exist for cleaned images
        for r in results:
            if r["status"] == "cleaned":
                name = os.path.basename(r["input"])
                assert os.path.exists(os.path.join(out, f"debug_{name}"))
                assert os.path.exists(os.path.join(out, f"mask_{name}"))

    def test_e2e_gemini(self, pipeline, real_gemini_image, output_dir):
        """E2E: process a real Gemini image and verify watermark removal."""
        out_path = os.path.join(output_dir, "clean_gemini.png")
        result_img, meta = pipeline.process(real_gemini_image, out_path)
        assert meta["status"] == "cleaned"
        assert meta["watermarks_found"] >= 1
        assert meta["methods"] == ["spatial_template_local_lama"]
        assert meta["total_detections"] >= 1
        assert meta["localization"]["accepted_regions"] == 1
        assert meta["localization"]["experts"] == [
            "spatial_template",
            "open_vocabulary",
        ]
        assert meta["regions"][0]["source"] == "spatial_template"
        assert os.path.exists(out_path)
        # Verify output is a valid image
        check = Image.open(out_path)
        assert check.size[0] > 0

    def test_e2e_doubao(self, pipeline, real_doubao_image, output_dir):
        """E2E: process a real Doubao image and verify watermark removal."""
        out_path = os.path.join(output_dir, "clean_doubao.jpg")
        result_img, meta = pipeline.process(real_doubao_image, out_path)
        assert meta["status"] == "cleaned"
        assert meta["watermarks_found"] >= 1
        assert os.path.exists(out_path)
