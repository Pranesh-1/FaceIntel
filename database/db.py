import sqlite3
import numpy as np  # type: ignore
import os
import logging
from database.models import CREATE_FACES_TABLE, CREATE_EVENTS_TABLE  # type: ignore

logger = logging.getLogger(__name__)


class Database:
    """
    Thread-safe SQLite wrapper using a new connection per operation
    (stateless design — no persistent connection held open).
    Call close() for a clean log message on shutdown.
    """
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute(CREATE_FACES_TABLE)
            conn.execute(CREATE_EVENTS_TABLE)
            conn.commit()
        logger.info("Database initialized.")

    def close(self):
        """No-op — connections are per-operation. Logs a clean shutdown message."""
        logger.info(f"Database '{self.db_path}' released.")

    # ── Face operations ──────────────────────────────────────────────────────

    def insert_face(self, face_id: str, embedding: np.ndarray):
        blob = embedding.astype(np.float32).tobytes()
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO faces (id, embedding) VALUES (?, ?)",
                (face_id, blob),
            )
            conn.commit()
        logger.debug(f"Face {face_id} saved to DB.")

    def get_all_faces(self) -> list[tuple[str, np.ndarray]]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT id, embedding FROM faces").fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    def count_unique_visitors(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM faces").fetchone()
        return row[0]

    # ── Event operations ─────────────────────────────────────────────────────

    def log_event(self, face_id: str, event_type: str, image_path: str | None = None):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO events (face_id, event_type, image_path) VALUES (?, ?, ?)",
                (face_id, event_type, image_path),
            )
            conn.commit()
        logger.debug(f"Event logged: {face_id} -> {event_type}")

    def get_events(self, face_id: str | None = None) -> list[dict]:
        with self._get_conn() as conn:
            if face_id:
                rows = conn.execute(
                    "SELECT id, face_id, event_type, timestamp, image_path "
                    "FROM events WHERE face_id = ? ORDER BY timestamp",
                    (face_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, face_id, event_type, timestamp, image_path "
                    "FROM events ORDER BY timestamp"
                ).fetchall()
        return [
            {
                "id": r[0],
                "face_id": r[1],
                "event_type": r[2],
                "timestamp": r[3],
                "image_path": r[4],
            }
            for r in rows
        ]

    def get_latest_events(self, limit: int = 12) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, face_id, event_type, timestamp, image_path "
                "FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()
        # Returns from newest to oldest
        return [
            {
                "id": r[0],
                "face_id": r[1],
                "event_type": r[2],
                "timestamp": r[3],
                "image_path": r[4],
            }
            for r in rows
        ]
    def clear_all(self):
        """Wipes all data from both tables."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM events")
            conn.execute("DELETE FROM faces")
            conn.commit()
        logger.info("Database wiped successfully.")
