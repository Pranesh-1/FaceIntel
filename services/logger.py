"""
Logger Service
Handles all event, file, and image logging for the face tracker.
- events.log: human-readable flat log of every critical system event.
- logs/entries/<date>/ and logs/exits/<date>/: dated image stores.
- Database logging is handled by db.py; this module calls it via a callback.
"""

import os
import logging
import logging.handlers
from datetime import datetime

from utils.image_utils import save_image, get_dated_dir  # type: ignore


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Configure the root logger with both a rotating file handler (events.log)
    and a console handler.  Call once at application startup.

    Returns the root logger so callers can retrieve it later with
    logging.getLogger(__name__).
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "events.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — rotates at 10 MB, keeps 5 backups
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    # Console handler — WARNING and above
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(message)s"))

    if not root.handlers:
        root.addHandler(fh)
        root.addHandler(ch)

    return root


class FaceEventLogger:
    """
    High-level logger that saves face images and records events.

    Args:
        db: Database instance for persisting events.
        log_dir: Root log directory (e.g. "logs").
    """

    def __init__(self, db, log_dir: str = "logs"):
        self.db = db
        self.log_dir = log_dir
        self._logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────────────────────────────────────
    def log_entry(self, face_id: str, face_crop) -> str | None:
        """Log a face ENTRY event: save crop and write to DB + log file."""
        return self._log_event(face_id, "ENTRY", face_crop)

    def log_exit(self, face_id: str, face_crop) -> str | None:
        """Log a face EXIT event: save crop and write to DB + log file."""
        return self._log_event(face_id, "EXIT", face_crop)

    def log_registration(self, face_id: str, face_crop) -> str | None:
        """Log a new face REGISTRATION event."""
        return self._log_event(face_id, "REGISTERED", face_crop)

    def log_recognition(self, face_id: str):
        """Log that an existing face was recognised (no image needed)."""
        self._logger.info(f"[RECOGNISED] {face_id}")
        self.db.log_event(face_id, "RECOGNISED")

    # ─────────────────────────────────────────────────────────────────────────
    def _log_event(self, face_id: str, event_type: str, face_crop) -> str | None:
        image_path = None
        if face_crop is not None and face_crop.size > 0:
            event_sub = "entries" if event_type.lower() == "entry" else event_type.lower() + "s"
            directory = get_dated_dir(self.log_dir, event_sub)
            ts = datetime.now().strftime("%H-%M-%S-%f")
            filename = f"{face_id}_{ts}.jpg"
            image_path = save_image(face_crop, directory, filename)

        self._logger.info(f"[{event_type}] {face_id} | image={image_path}")
        self.db.log_event(face_id, event_type, image_path)
        return image_path
