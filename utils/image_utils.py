import cv2  # type: ignore
import numpy as np  # type: ignore
import os
from datetime import datetime


def crop_face(frame: np.ndarray, bbox: list | tuple, padding: int = 40) -> np.ndarray:
    """
    Crop the face region from a YOLO *person* bounding box.
    Extracts only the top 40% of the box (head/shoulders) to drastically
    reduce RetinaFace pixel parsing overhead and latency.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    person_h = y2 - y1
    y2_head = y1 + int(person_h * 0.40)
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w_frame, x2 + padding)
    y2 = min(h_frame, y2_head + padding)
    
    return frame[y1:y2, x1:x2].copy()


def save_image(image: np.ndarray, directory: str, filename: str) -> str:
    """Save an image to disk, creating directories as needed. Returns the saved path."""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    cv2.imwrite(filepath, image)
    return filepath


def get_dated_dir(base_dir: str, sub: str) -> str:
    """Return a date-bucketed directory path like base_dir/sub/YYYY-MM-DD/."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(base_dir, sub, date_str)
