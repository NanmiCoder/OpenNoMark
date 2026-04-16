"""Watermark detection using OWLv2 open-vocabulary object detection."""

import torch
import numpy as np


class WatermarkDetector:
    def __init__(self, device=None, score_threshold=0.05, corner_ratio=0.15, max_size_ratio=0.30):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device
        self.score_threshold = score_threshold
        self.corner_ratio = corner_ratio
        self.max_size_ratio = max_size_ratio

        model_id = "google/owlv2-base-patch16-ensemble"
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device)

    def detect(self, image):
        """Detect watermark candidates in image. Returns list of {box, label, score}."""
        text_queries = [["watermark", "logo", "icon", "symbol", "badge", "stamp"]]

        all_boxes = []
        for queries in text_queries:
            inputs = self.processor(text=queries, images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=self.score_threshold
            )

            result = results[0]
            boxes = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = result["labels"].cpu().numpy()

            for box, score, label_idx in zip(boxes, scores, labels):
                all_boxes.append({
                    "box": box.tolist(),
                    "label": queries[label_idx],
                    "score": float(score),
                })

        return all_boxes

    def filter_watermarks(self, boxes, image_width, image_height):
        """Keep only small marks in extreme corners."""
        max_size = min(image_width, image_height) * self.max_size_ratio
        cr = self.corner_ratio

        filtered = []
        for item in boxes:
            x1, y1, x2, y2 = item["box"]
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if w > max_size or h > max_size:
                continue

            in_corner = (
                (cx > image_width * (1 - cr) or cx < image_width * cr)
                and (cy > image_height * (1 - cr) or cy < image_height * cr)
            )
            if in_corner:
                filtered.append(item)

        return filtered
