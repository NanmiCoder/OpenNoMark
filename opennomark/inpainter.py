"""LaMa-based inpainting with feathered mask blending."""

import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw


def _ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


class LamaInpainter:
    def __init__(self, device=None):
        model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/big-lama.pt")
        if not os.path.exists(model_path):
            from torch.hub import download_url_to_file
            url = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            download_url_to_file(url, model_path)

        # Device selection: CUDA when available, otherwise CPU. MPS is
        # intentionally skipped — LaMa's TorchScript graph contains ops
        # (e.g. FFT variants) that are not supported on Apple MPS, so we
        # fall back to CPU on Mac instead of producing garbage output.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, torch.device):
            device = device.type
        if device == "mps":
            device = "cpu"
        self.device = torch.device(device)

        # The checkpoint is CUDA-serialized; deserialize on CPU first, then
        # move to the target device. This is what makes it loadable on
        # CPU-only and non-NVIDIA machines.
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

    def create_mask(self, image_size, boxes, padding=3, feather=4):
        """Create a feathered mask from bounding boxes.

        Defaults chosen to minimize bleed across structural edges: large
        padding + heavy feather causes LaMa to 'helpfully' fill in across
        high-contrast boundaries (e.g. paint white fabric over a black
        panel adjacent to a sparkle). Tight values keep LaMa focused on
        the watermark pixels themselves.
        """
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)

        for item in boxes:
            x1, y1, x2, y2 = item["box"]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image_size[0], x2 + padding)
            y2 = min(image_size[1], y2 + padding)
            draw.rectangle([x1, y1, x2, y2], fill=255)

        if feather > 0:
            mask_np = np.array(mask)
            mask_np = cv2.GaussianBlur(mask_np, (0, 0), sigmaX=feather)
            if mask_np.max() > 0:
                mask_np = np.clip(mask_np.astype(np.float32) / mask_np.max() * 255, 0, 255).astype(np.uint8)
            mask = Image.fromarray(mask_np)

        return mask

    def inpaint(self, image, mask):
        """Run LaMa inpainting with alpha blending."""
        orig_w, orig_h = image.size
        orig_np = np.array(image).astype(np.float32)

        img_np = orig_np / 255.0
        img_np = np.transpose(img_np, (2, 0, 1))

        mask_np = np.array(mask).astype(np.float32) / 255.0
        hard_mask = (mask_np > 0.1).astype(np.float32)[np.newaxis, ...]

        _, h, w = img_np.shape
        pad_h = _ceil_modulo(h, 8) - h
        pad_w = _ceil_modulo(w, 8) - w
        img_padded = np.pad(img_np, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")
        mask_padded = np.pad(hard_mask, ((0, 0), (0, pad_h), (0, pad_w)), mode="symmetric")

        img_t = torch.from_numpy(img_padded).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_padded).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            out = self.model(img_t, mask_t)

        inpainted_np = out[0].permute(1, 2, 0).cpu().numpy()
        inpainted_np = inpainted_np[:orig_h, :orig_w, :]
        inpainted_np = np.clip(inpainted_np * 255, 0, 255)

        alpha = mask_np[:, :, np.newaxis]
        blended = inpainted_np * alpha + orig_np * (1.0 - alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return Image.fromarray(blended)
