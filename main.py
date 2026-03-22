"""
main.py — Face Tracker Entry Point
Orchestrates the complete face detection, tracking, recognition, and logging pipeline.
"""

import os
# 🔥 ULTRA CLEAN MODE - Mute all 3rd party internal logs BEFORE they load
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["INSIGHTFACE_LOG_LEVEL"] = "ERROR"

import argparse
import logging
import time
import cv2  # type: ignore

import warnings
warnings.filterwarnings("ignore")

from utils.config_loader import ConfigLoader  # type: ignore

from core.tracker import FaceTracker  # type: ignore
from core.embedding import EmbeddingGenerator  # type: ignore
from core.matcher import FaceMatcher  # type: ignore

from database.db import Database  # type: ignore

from services.logger import setup_logging, FaceEventLogger  # type: ignore
from services.face_registry import FaceRegistry  # type: ignore
from services.visitor_counter import VisitorCounter  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description="Intelligent Face Tracker")
    parser.add_argument("--source", type=str, default=None,
                        help="Video file path, RTSP URL, or camera index (overrides config).")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to config.json.")
    parser.add_argument("--show", action="store_true",
                        help="Display live annotated video window.")
    parser.add_argument("--clear-db", action="store_true",
                        help="Clear the existing database before starting (starts count at 0).")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = ConfigLoader(args.config)
    log_dir        = cfg.get("log_dir", "logs")
    db_path        = cfg.get("db_path", "database/face_tracker.db")
    # 🔥 ELITE HYBRID RECOVERY: frame_skip=2 for fluid 30 FPS playback
    frame_skip     = 2
    sim_threshold  = float(cfg.get("similarity_threshold", 0.45))
    exit_threshold = int(cfg.get("exit_frame_threshold", 30))

    # ── Logging setup ────────────────────────────────────────────────────────
    setup_logging(log_dir)
    logger = logging.getLogger("main")
    logger.info("=== Face Tracker Starting ===")

    # ── Resolve video source ─────────────────────────────────────────────────
    source = args.source or cfg.get("camera_source", "0")
    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    logger.info(f"Source opened — FPS={fps}, frame_skip={frame_skip}")
    
    # 🔥 CLI CLEANUP (PRO MODE)
    print("🚀 Face Tracker Started")
    print(f"📷 Source: {source}")
    print("🧠 Model: InsightFace (CPU)")
    print("-----------------------------------")

    # ── Component Initialisation ─────────────────────────────────────────────
    db = Database(db_path)
    if args.clear_db:
        db.clear_all()
        logger.info("Database cleared as requested.")

    embedding_gen = EmbeddingGenerator()
    matcher = FaceMatcher(sim_threshold)
    event_logger = FaceEventLogger(db, log_dir)
    visitor_counter = VisitorCounter(db)
    
    # 🏁 FAST HYBRID PIPELINE: YOLO-Body + RetinaFace-Crops
    # The Hybrid Tracking Engine: YOLOv8 for coarse body tracking + InsightFace for identity
    tracker = FaceTracker(conf_threshold=0.40)
    registry = FaceRegistry(db, embedding_gen, matcher, event_logger, visitor_counter)

    # ── Main Processing Loop ─────────────────────────────────────────────────
    frame_number = 0
    session_unique_ids = set() 
    prev_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream.")
                break

            frame_number += 1
            if frame_number % frame_skip != 0:
                continue

            # 1. FAST BODY TRACKING (YOLOv8)
            tracked_results = tracker.track(frame)
            
            # 2. HYBRID FACE PROCESSING (RetinaFace + Elite Registry)
            seen_this_frame = registry.process_detections(frame, frame_number, tracked_results)
            
            # 3. HUD DATA & TELEMETRY
            current_time = time.time()
            fps_val = 1.0 / max((current_time - prev_time), 0.001)
            prev_time = current_time

            global_count = visitor_counter.count
            for t in tracked_results:
                fid = registry.get_face_id(t["track_id"])
                if fid: session_unique_ids.add(fid)

            # ── [OPTIONAL] Visualization ─────────────────────────────────────────
            if args.show:
                annotated = frame.copy()
                for t in tracked_results:
                    tid = t["track_id"]
                    bbox = [int(v) for v in t["bbox"]]
                    fid = registry.get_face_id(tid)
                    
                    if fid is None:
                        colour = (0, 0, 255) # Red for NEW FACE
                        status_label = "[NEW FACE]"
                        display_id = "Analyzing..."
                    else:
                        colour = (0, 255, 0) # Green for RECOGNIZED
                        status_label = "[RECOGNIZED]"
                        display_id = str(fid)
                    
                    label = f"{status_label} ID: {display_id} | Track: {tid}"
                    cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colour, 2)
                    cv2.putText(annotated, label, (bbox[0], bbox[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2)

                # 🔥 HIGH-VISIBILITY HUD (The 'Unique Count' Fix)
                cv2.rectangle(annotated, (0, 0), (350, 150), (0, 0, 0), -1) # Black background
                cv2.putText(annotated, f"Visitors: {global_count}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                cv2.putText(annotated, f"Session Count: {len(session_unique_ids)}", (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated, f"FPS: {int(fps_val)}", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 🔥 IMMERSIVE FULL-SCREEN HUD
                cv2.namedWindow("Face Tracker", cv2.WINDOW_NORMAL)
                cv2.setWindowProperty("Face Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                
                cv2.imshow("Face Tracker", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        db.close()
        logger.info(f"=== Session complete | Global DB Visitors: {visitor_counter.count} ===")

    print("\n-----------------------------------")
    print("✅ Session Complete")
    print(f"👥 Total Visitors: {visitor_counter.count}")


if __name__ == "__main__":
    # Ensure current directory is in path
    import sys
    sys.path.append(os.getcwd())
    main()
