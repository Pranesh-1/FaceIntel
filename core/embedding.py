"""
Embedding Module
Uses InsightFace (ArcFace) to generate 512-dimensional facial embeddings.
"""

import numpy as np  # type: ignore
import logging
import cv2  # type: ignore
import sys
import os

logger = logging.getLogger(__name__)

class HiddenPrints:
    def __init__(self):
        self._original_stdout = sys.stdout

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class EmbeddingGenerator:
    """
    Wraps InsightFace ArcFace for SOTA 512-d embeddings.
    """

    def __init__(self):
        import insightface  # type: ignore
        from insightface.app import FaceAnalysis  # type: ignore

        logger.info("Loading InsightFace model (buffalo_l: RetinaFace + ArcFace)...")
        # Initialize FaceAnalysis.
        with HiddenPrints():
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            # ctx_id=-1 explicitly requests CPU
            self.app.prepare(ctx_id=-1, det_size=(640, 640))
        logger.info("InsightFace ArcFace model loaded successfully.")

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Detect all high-quality faces in a full frame.
        Filters: det_score >= 0.5, size >= 40px.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            faces = self.app.get(frame)
            valid_faces = []
            for face in faces:
                # 🛡️ THE GOLDEN THRESHOLD: 0.4 as per Final Config (Excellent Recall)
                if getattr(face, 'det_score', 0.0) < 0.4:
                    continue

                # 🛡️ THE GOLDEN THRESHOLD: 40px minimum size for distant subjects
                x1, y1, x2, y2 = getattr(face, 'bbox', [0,0,0,0])
                if (x2 - x1) < 40 or (y2 - y1) < 40:
                    continue

                valid_faces.append(face)
            
            return valid_faces
        except Exception as e:
            logger.error(f"Full-frame detection error: {e}")
            return []

    def get_embedding(self, image: np.ndarray) -> tuple:
        """
        Extract a normalised 512-d embedding and face bbox from an image.
        Returns: (embedding, quality_score, face_bbox)
        """
        if image is None or image.size == 0:
            return None, 0.0, None

        try:
            # insightface expects BGR image (native OpenCV format)
            faces = self.app.get(image)
            
            if not faces:
                return None, 0.0, None
                
            # Filter by detection confidence and size.
            valid_faces = [
                f for f in faces 
                if getattr(f, 'det_score', 0.0) >= 0.40 # Universal baseline for Person crops
                and (getattr(f, 'bbox', [0,0,0,0])[2] - getattr(f, 'bbox', [0,0,0,0])[0]) >= 15 # Smallest detectable face
            ]
            
            if not valid_faces:
                return None, 0.0, None
                
            # If multiple valid faces, take the largest one
            def face_area(f):
                bbox = getattr(f, 'bbox', [0,0,0,0])
                return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            best_face = max(valid_faces, key=face_area)
            
            # ELITE GATE: For spawning a NEW IDENTITY, we accept 0.45+ det_score
            # because we require a 10-frame persistent sequence in FaceRegistry.
            is_high_quality = (getattr(best_face, 'det_score', 0.0) >= 0.45)
            
            # normed_embedding check (Verified SOTA L2-norm gate)
            emb = getattr(best_face, 'normed_embedding', None)
            if emb is None or np.linalg.norm(emb) < 0.85:
                return None, 0.0, None
            
            # Return tuple: (embedding, is_high_quality, face_bbox)
            face_bbox = getattr(best_face, 'bbox', None)
            return (np.array(emb, dtype=np.float32), is_high_quality, face_bbox)

        except Exception as e:
            logger.debug(f"InsightFace embedding error: {e}")
            return None, 0.0, None

    def _get_empty_return(self):
        return None, 0.0, None
