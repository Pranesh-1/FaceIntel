"""
Face Registry Service
Central registration and tracking-memory hub.
"""

import uuid
import logging
import numpy as np  # type: ignore

logger = logging.getLogger(__name__)


class FaceRegistry:
    """
    Manages face identity across frames with Hybrid Association (Person -> Face).
    """

    def __init__(self, db, embedding_gen, matcher, event_logger, visitor_counter):
        self.db = db
        self.embedding_gen = embedding_gen
        self.matcher = matcher
        self.event_logger = event_logger
        self.visitor_counter = visitor_counter

        # Load registered gallery
        self._registered: dict[str, np.ndarray] = {
            fid: emb for fid, emb in db.get_all_faces()
        }
        self._track_to_face: dict[int, str] = {} # track_id -> face_id
        self._face_last_seen: dict[str, int] = {fid: 0 for fid in self._registered}
        self._inside_ids: set[str] = set()
        
        # 🛡️ RECALL UPGRADE: Temporal "Face Hold" (Flicker Immunity)
        self._face_hold: dict[int, dict] = {}
        
        # Validation for new identities
        self._validation_counts: dict[int, dict] = {}
        self.val_frames = 5
        self.inference_skip = 15

        # 🛡️ ASYNC THREADING STATE
        from threading import Lock
        self._async_lock = Lock()
        self._processing_tids: set[int] = set()
        self._last_processed: dict[int, int] = {}

        # 🛡️ UI METRICS EXPORT
        self._track_info: dict[int, dict] = {}

        logger.info(
            f"FaceRegistry loaded {len(self._registered)} face(s) from database."
        )

    def process_detections(
        self, frame: np.ndarray, frame_number: int, tracked_results: list[dict]
    ) -> set[int]:
        """
        Ultra-Fast Hybrid Association (Person -> Face-Crop).
        """
        seen_ids_this_frame: set[int] = set()
        from utils.image_utils import crop_face # type: ignore
        from threading import Thread

        for track in tracked_results:
            tid = track["track_id"]
            seen_ids_this_frame.add(tid)
            bbox = track["bbox"]
            
            # 🔥 SMART ADAPTIVE INFERENCE
            # We never re-identify established tracks to sustain max FPS.
            existing_id = self._track_to_face.get(tid)
            
            if existing_id is not None:
                self._face_last_seen[existing_id] = frame_number
                continue
                
            # If already processing async, let it work
            if tid in self._processing_tids:
                continue
                
            # Cooldown logic (Don't flood threads every frame for the same track)
            last_p = self._last_processed.get(tid, 0)
            if frame_number - last_p < 5:
                continue

            # 1. ASYNC OPTIMIZED INFERENCE
            face_crop = crop_face(frame, bbox)
            if face_crop.size > 0:
                self._processing_tids.add(tid)
                self._last_processed[tid] = frame_number
                Thread(target=self._async_extract, args=(tid, face_crop, frame_number, bbox), daemon=True).start()

        # 4. Cleanup expired holds
        expired = [tid for tid, h in self._face_hold.items() if tid not in seen_ids_this_frame and h["ttl"] <= 0]
        for tid in expired: 
            self._face_hold.pop(tid, None)
            self._track_info.pop(tid, None)

        return seen_ids_this_frame

    def _async_extract(self, tid: int, face_crop: np.ndarray, frame_number: int, bbox: list):
        try:
            embedding, quality, face_bbox = self.embedding_gen.get_embedding(face_crop)
            
            reused = False
            if embedding is not None:
                self._face_hold[tid] = {"embedding": embedding, "ttl": 15, "quality": quality}
            elif tid in self._face_hold and self._face_hold[tid]["ttl"] > 0:
                embedding = self._face_hold[tid]["embedding"]
                quality = self._face_hold[tid]["quality"]
                self._face_hold[tid]["ttl"] -= 1
                reused = True

            with self._async_lock:
                if embedding is not None:
                    pass_quality = 0.0 if reused else quality
                    self._handle_identity(tid, embedding, face_crop, frame_number, pass_quality, face_bbox, bbox)
                else:
                    self._handle_identity(tid, None, None, frame_number, 0.0, None, bbox)
        finally:
            self._processing_tids.discard(tid)

    def _handle_identity(self, track_id: int, embedding: np.ndarray | None, face_crop: np.ndarray | None, frame_number: int, quality: float, face_bbox: list | None = None, body_bbox: list | None = None):
        """Processes identity with THE ELITE LOCK."""
        # 🛡️ THE ELITE IDENTITY LOCK
        existing_id = self._track_to_face.get(track_id)
        if existing_id:
            self._face_last_seen[existing_id] = frame_number
            # Predict smooth face box movement while skipping inference
            if track_id in self._track_info and "relative_face_box" in self._track_info[track_id] and body_bbox:
                rx1, ry1, rx2, ry2 = self._track_info[track_id]["relative_face_box"]
                bx1, by1, _, _ = body_bbox
                self._track_info[track_id]["face_bbox"] = [bx1+rx1, by1+ry1, bx1+rx2, by1+ry2]
            return
        
        if embedding is None:
            return

        # Match check
        best_id, similarity = self.matcher.match(embedding, list(self._registered.items()))

        # Save metrics for UI
        if track_id not in self._track_info: self._track_info[track_id] = {}
        self._track_info[track_id]["sim"] = float(similarity)
        
        if face_bbox is not None and body_bbox is not None:
            bx1, by1, _, _ = body_bbox
            fx1, fy1, fx2, fy2 = face_bbox
            self._track_info[track_id]["relative_face_box"] = [fx1, fy1, fx2, fy2]
            self._track_info[track_id]["face_bbox"] = [
                int(bx1 + fx1), int(by1 + fy1), int(bx1 + fx2), int(by1 + fy2)
            ]

        # Match priority
        if best_id and similarity >= self.matcher.threshold:
            self._track_to_face[track_id] = best_id
            self._face_last_seen[best_id] = frame_number
            if best_id not in self._inside_ids:
                self.event_logger.log_recognition(best_id)
                self.event_logger.log_entry(best_id, face_crop)
                self._inside_ids.add(best_id)
            self._validation_counts.pop(track_id, None)
            return

        # Registration (0.72 Floor OR 0.55 Temporal Validation)
        if quality >= 0.55:
            stats = self._validation_counts.get(track_id, {"count": 0, "sum_q": 0.0})
            stats["count"] += 1
            stats["sum_q"] += quality
            self._validation_counts[track_id] = stats
            
            avg_q = stats["sum_q"] / stats["count"]
            if quality >= 0.72 or (stats["count"] >= self.val_frames and avg_q >= 0.55):
                face_id = self._generate_id()
                self._registered[face_id] = embedding
                self._face_last_seen[face_id] = frame_number
                self._track_to_face[track_id] = face_id
                self.db.insert_face(face_id, embedding)
                self.visitor_counter.increment()
                self.event_logger.log_registration(face_id, face_crop)
                self.event_logger.log_entry(face_id, face_crop)
                self._inside_ids.add(face_id)
                print(f"\n👤 NEW FACE → ID: {face_id}")
                print(f"📈 Visitors: {self.visitor_counter.count}")
                logger.info(f"NEW FACE REGISTRATION: track={track_id} -> {face_id} (det_score={avg_q:.3f})")
                self._validation_counts.pop(track_id, None)

    def check_exits(self, seen_this_frame: set[int], frame_number: int, exit_threshold: int, frame_getter):
        """Standard exit detection."""
        for tid in list(self._track_to_face.keys()):
            fid = self._track_to_face.get(tid)
            last_seen = self._face_last_seen.get(fid, 0) if fid else 0
            # If the track hasn't been active in recent frames past the threshold, officially log an exit.
            if tid not in seen_this_frame and (frame_number - last_seen > exit_threshold):
                face_id = self._track_to_face.pop(tid, None)
                if face_id:
                    self.event_logger.log_exit(face_id, None)
                    self._inside_ids.discard(face_id)
                    
    def get_face_id(self, track_id: int) -> str | None:
        return self._track_to_face.get(track_id)

    def is_new_registration(self, track_id: int) -> bool:
        return track_id in self._validation_counts # Simple proxy

    def _generate_id(self) -> str:
        guid = str(uuid.uuid4().hex)
        return f"face_{guid[0:6]}" # type: ignore
