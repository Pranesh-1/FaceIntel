"""
Face Tracker Module
Implements a lightweight IoU-based tracker that wraps YOLO's built-in
ByteTrack integration via ultralytics.
"""

import numpy as np  # type: ignore
import logging
from ultralytics import YOLO  # type: ignore

logger = logging.getLogger(__name__)


class FaceTracker:
    """
    Wraps YOLO's .track() API (ByteTrack under the hood) to deliver
    persistent track IDs across frames.
    """

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.40):
        logger.info(f"Initialising FaceTracker with model '{model_path}' and conf={conf_threshold}.")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track(self, frame: np.ndarray) -> list[dict]:
        """
        Run tracking on a frame.

        Returns:
            List of dicts: {"track_id": int, "bbox": [x1,y1,x2,y2], "conf": float}
        """
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=self.conf_threshold,
            classes=[0],  # 0 indicates person
            tracker="bytetrack.yaml",
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if box.id is None:
                    # Unconfirmed track — skip
                    continue
                track_id = int(box.id[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append(
                    {"track_id": track_id, "bbox": [x1, y1, x2, y2], "conf": conf}
                )
        return detections
