"""Unit tests for LaMa inpainter."""

import numpy as np
import pytest
from PIL import Image


class TestLamaInpainter:
    """Test LaMa-based inpainting."""

    @pytest.fixture(scope="class")
    def inpainter(self):
        from opennomark.inpainter import LamaInpainter
        return LamaInpainter()

    def test_init(self, inpainter):
        assert inpainter.model is not None

    def test_create_mask_basic(self, inpainter):
        boxes = [{"box": [100, 100, 150, 150]}]
        mask = inpainter.create_mask((400, 400), boxes, padding=10, feather=0)
        mask_np = np.array(mask)
        assert mask_np.shape == (400, 400)
        # Area inside box should be white
        assert mask_np[120, 120] == 255
        # Area far from box should be black
        assert mask_np[300, 300] == 0

    def test_create_mask_feathered(self, inpainter):
        boxes = [{"box": [100, 100, 150, 150]}]
        mask = inpainter.create_mask((400, 400), boxes, padding=10, feather=12)
        mask_np = np.array(mask)
        # Core should still be bright
        assert mask_np[125, 125] > 200
        # Edge should be feathered (gradient, not hard 0/255)
        edge_val = mask_np[80, 125]
        assert 0 < edge_val < 255, f"Edge should be feathered, got {edge_val}"

    def test_create_mask_multiple_boxes(self, inpainter):
        boxes = [
            {"box": [10, 10, 50, 50]},
            {"box": [300, 300, 350, 350]},
        ]
        mask = inpainter.create_mask((400, 400), boxes, padding=5, feather=0)
        mask_np = np.array(mask)
        assert mask_np[30, 30] == 255
        assert mask_np[325, 325] == 255
        assert mask_np[200, 200] == 0

    def test_inpaint_preserves_size(self, inpainter):
        img = Image.new("RGB", (256, 256), color=(100, 100, 100))
        mask = Image.new("L", (256, 256), 0)
        from PIL import ImageDraw
        draw = ImageDraw.Draw(mask)
        draw.rectangle([100, 100, 150, 150], fill=255)

        result = inpainter.inpaint(img, mask)
        assert result.size == img.size
        assert result.mode == "RGB"

    def test_inpaint_modifies_masked_area(self, inpainter):
        # Create image with a white square on black background
        img = Image.new("RGB", (128, 128), color=(0, 0, 0))
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 80, 80], fill=(255, 255, 255))

        # Mask the white square
        mask = Image.new("L", (128, 128), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.rectangle([48, 48, 82, 82], fill=255)

        result = inpainter.inpaint(img, mask)
        result_np = np.array(result)
        orig_np = np.array(img)

        # The masked region should be different from original
        masked_region_orig = orig_np[50:80, 50:80].mean()
        masked_region_result = result_np[50:80, 50:80].mean()
        assert masked_region_orig != masked_region_result, "Inpainting should modify masked area"

    def test_inpaint_zero_mask_preserves_image(self, inpainter):
        img = Image.new("RGB", (128, 128), color=(42, 100, 200))
        mask = Image.new("L", (128, 128), 0)  # empty mask

        result = inpainter.inpaint(img, mask)
        # With zero mask + alpha blending, result should equal original
        diff = np.abs(np.array(result).astype(float) - np.array(img).astype(float)).mean()
        assert diff < 2.0, f"Empty mask should preserve original, diff={diff}"
