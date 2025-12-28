from ultralytics import YOLO
import cv2
import numpy as np
from .config import MODEL_PATH, CLASS_NAMES


class ParkingAttendantDetector:
    def __init__(self, model_path: str = MODEL_PATH, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = CLASS_NAMES

    def predict(self, image_bgr: np.ndarray):
        """
        Run detection on a single BGR image (OpenCV format).

        Returns a list of detections:
        [
            {
                "cls": int,            # class index
                "label": str,          # class name
                "conf": float,         # confidence [0,1]
                "bbox": [x1,y1,x2,y2]  # int coordinates
            },
            ...
        ]
        """
        results = self.model(image_bgr, verbose=False)[0]

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue

            cls_idx = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "cls": cls_idx,
                "label": self.class_names[cls_idx],
                "conf": conf,
                "bbox": [x1, y1, x2, y2],
            })

        return detections

    def draw_detections(self, image_bgr: np.ndarray, detections):
        """
        Returns a copy of the image with bounding boxes and labels drawn.
        Parking attendants are green; others are red.
        """
        out = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            conf = det["conf"]

            if det["cls"] == 1:  # parking_attendant
                color = (0, 255, 0)  # green
            else:
                color = (0, 0, 255)  # red

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            cv2.putText(out, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out
