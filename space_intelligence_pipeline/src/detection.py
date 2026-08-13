"""
detection.py — Real crater detection using fine-tuned YOLOv8 model.
Replaces the placeholder. Loads local weights and returns structured detections.
"""

import os
from ultralytics import YOLO


class CraterDetector:
    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        """
        model_path: path to trained .pt weights. Defaults to the bundled
                    crater_best_v2.pt shipped in src/weights/.
        conf_threshold: minimum confidence to keep a detection.
        """
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(__file__), "weights", "crater_best_v2.pt"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Crater detection weights not found at {model_path}. "
                f"Make sure crater_best_v2.pt is committed to src/weights/."
            )

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image_path: str, save_dir: str = None) -> dict:
        """
        Run detection on a single image.

        Returns:
            {
                "annotated_image_path": str,
                "detections": [
                    {"label": str, "confidence": float, "box": [x1, y1, x2, y2]},
                    ...
                ],
                "count": int
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        results = self.model.predict(
            image_path,
            conf=self.conf_threshold,
            save=True,
            project=save_dir if save_dir else "runs/detect",
            name="predict",
            exist_ok=True,
        )

        r = results[0]
        detections = []
        for box in r.boxes:
            label = self.model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            coords = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": round(confidence, 3),
                "box": [round(c, 1) for c in coords],
            })

        annotated_path = os.path.join(r.save_dir, os.path.basename(image_path))

        return {
            "annotated_image_path": annotated_path,
            "detections": detections,
            "count": len(detections),
        }


if __name__ == "__main__":
    # Quick manual test
    detector = CraterDetector()
    result = detector.detect("path/to/test/image.jpg")
    print(f"Detected {result['count']} crater(s)")
    for d in result["detections"]:
        print(d)
