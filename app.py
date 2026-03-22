#!/usr/bin/env python3
import os
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["INSIGHTFACE_LOG_LEVEL"] = "ERROR"

import time
import threading
import copy
import cv2  # type: ignore
import logging
import warnings
import typing
import numpy as np  # type: ignore
import datetime
warnings.filterwarnings("ignore")

from flask import Flask, render_template, Response, jsonify, send_from_directory  # type: ignore[import]

from utils.config_loader import ConfigLoader  # type: ignore
from core.tracker import FaceTracker  # type: ignore
from core.embedding import EmbeddingGenerator  # type: ignore
from core.matcher import FaceMatcher  # type: ignore
from database.db import Database  # type: ignore
from services.logger import setup_logging, FaceEventLogger  # type: ignore
from services.face_registry import FaceRegistry  # type: ignore
from services.visitor_counter import VisitorCounter  # type: ignore

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Global state for web sharing
lock = threading.Lock()
locked_frame: typing.Any = None
locked_stats: dict[str, typing.Any] = {
    "fps": 0,
    "global_count": 0,
    "session_count": 0,
    "faces": [],
    "events": []
}
config_lock = threading.Lock()
active_source: typing.Any = None
trigger_reset: bool = False

def tracker_loop():
    global locked_frame, locked_stats, active_source, trigger_reset
    cfg = ConfigLoader("config.json")
    db_path = cfg.get("db_path", "database/face_tracker.db")
    frame_skip = 2
    sim_threshold = float(cfg.get("similarity_threshold", 0.45))
    exit_threshold = int(cfg.get("exit_frame_threshold", 30))

    setup_logging("logs")
    db = Database(db_path)
    embedding_gen = EmbeddingGenerator()
    matcher = FaceMatcher(sim_threshold)
    event_logger = FaceEventLogger(db, "logs")
    visitor_counter = VisitorCounter(db)

    # Hybrid tracking enabled: YOLO (coarse body tracking) + InsightFace (identity)
    tracker = FaceTracker(conf_threshold=0.40)
    registry = FaceRegistry(db, embedding_gen, matcher, event_logger, visitor_counter)

    cap: typing.Any = None
    frame_number: int = 0
    session_unique_ids: set = set()
    tracks_history: "dict[int, list]" = {}  # Pyre2: use string annotation
    prev_time = float(time.perf_counter())
    fps_val: float = 0.0
    last_inference_ms: float = 0.0
    global_count = visitor_counter.count

    # Initiate default source
    source = cfg.get("camera_source", "0")
    if source.isdigit(): source = int(source)
    
    active_source = source

    while True:
        with config_lock:
            pending_source = active_source
            pending_reset = trigger_reset
            if pending_source is not None:
                active_source = None
            if pending_reset:
                trigger_reset = False

        if pending_source is not None:
            if cap is not None:
                cap.release()  # type: ignore
                cap = None
            with lock:
                locked_frame = None  # Clear stale frame so browser shows black
            if isinstance(pending_source, int) or str(pending_source) == "0":
                cap = cv2.VideoCapture(int(pending_source), cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(pending_source)
            print(f"\n🎥 Switched to: {pending_source}")

        if pending_reset:
            db.clear_all()
            tracks_history.clear()
            # Lightweight reset — clear internal registry state without reloading InsightFace
            registry._track_to_face.clear()  # type: ignore
            registry._face_last_seen.clear()  # type: ignore
            registry._inside_ids.clear()  # type: ignore
            registry._face_hold.clear()  # type: ignore
            registry._track_info.clear()  # type: ignore
            registry._validation_counts.clear()  # type: ignore
            registry._processing_tids.clear()  # type: ignore
            registry._last_processed.clear()  # type: ignore
            registry._registered.clear()  # type: ignore
            visitor_counter._count = 0  # type: ignore  # count is @property; reset private attr
            global_count = 0
            session_unique_ids.clear()
            with lock:
                locked_stats["global_count"] = 0
                locked_stats["session_count"] = 0
                locked_stats["events"] = []
                locked_stats["faces"] = []
            frame_number = 0
            last_inference_ms = 0.0
            print("🧹 Reset Complete.")

        if cap is None or not cap.isOpened():  # type: ignore
            time.sleep(0.1)
            continue

        ret, frame = cap.read()  # type: ignore
        if not ret or frame is None:
            # Video ended. Stop looping and wait for new selection!
            cap.release()  # type: ignore
            cap = None
            continue

        frame_number += 1  # type: ignore
        if int(frame_number) % int(frame_skip) != 0:
            continue

        # Periodic status heartbeat (professional requirement)
        if int(frame_number) % 100 == 0:
            logger.info(f"System Operational | Frame: {frame_number} | Unique IDs: {len(registry._registered)}")

        t_start = float(time.perf_counter())
        tracked_results = tracker.track(frame)
        seen_this_frame = registry.process_detections(frame, frame_number, tracked_results)
        registry.check_exits(seen_this_frame, frame_number, exit_threshold, None)
        curr_latency = max((float(time.perf_counter()) - t_start) * 1000.0, 1.0)  # type: ignore
        if last_inference_ms == 0:
            last_inference_ms = curr_latency
        else:
            last_inference_ms = 0.9 * last_inference_ms + 0.1 * curr_latency

        current_time = float(time.perf_counter())
        time_diff = max(current_time - prev_time, 0.001)  # type: ignore
        inst_fps: float = float(frame_skip) / time_diff
        fps_prev: float = fps_val
        fps_val = (0.8 * fps_prev + 0.2 * inst_fps) if fps_prev > 0.0 else inst_fps  # type: ignore[operator]
        prev_time = current_time

        global_count = visitor_counter.count  # type: ignore
        current_faces = []

        annotated = frame.copy()

        # Build trail history and cleanup old trails
        active_tids = {t["track_id"] for t in tracked_results}
        for tid in list(tracks_history.keys()):
            if tid not in active_tids:
                tracks_history.pop(tid, None)

        for t in tracked_results:
            tid = t["track_id"]
            bbox = [int(v) for v in t["bbox"]]
            fid = registry.get_face_id(tid)  # type: ignore
            info = registry._track_info.get(tid, {})  # type: ignore
            sim = info.get("sim", 0.0)
            face_box = info.get("face_bbox", None)
            val_count = registry._validation_counts.get(tid, {}).get("count", 0)  # type: ignore
            
            # Trail Logic
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            if tid not in tracks_history:
                tracks_history[tid] = []  # type: ignore[assignment]
            history: list = tracks_history[tid]  # type: ignore[assignment]
            history.append((center_x, center_y))
            if len(history) > 20:
                history.pop(0)

            # Draw Face Box (Cyan)
            if face_box:
                fb = [int(v) for v in face_box]
                cv2.rectangle(annotated, (fb[0], fb[1]), (fb[2], fb[3]), (255, 255, 0), 1)
            
            if fid:
                session_unique_ids.add(fid)
                # Gradient Color: Green (>0.65), Yellow (>0.50), Red (<0.50)
                if sim >= 0.65:
                    colour = (0, 255, 0)
                    hex_color = "#00ff00"
                elif sim >= 0.50:
                    colour = (0, 255, 255)
                    hex_color = "#ffff00"
                else:
                    colour = (0, 165, 255)
                    hex_color = "#ffa500"
                    
                status_label = "[RECOGNIZED]"
                display_id = str(fid)
                label = f"{status_label} ID: {display_id} | Sim: {sim:.2f} | Trk: {tid}"
            else:
                colour = (0, 0, 255)
                hex_color = "#ff3333"
                status_label = "[NEW FACE]"
                display_id = "Analyzing..."
                label = f"{status_label} Analyzing ({val_count}/5) | Trk: {tid}"
                
            # Draw trail
            history = tracks_history.get(tid, [])
            if len(history) > 1:
                pts = np.array(history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated, [pts], False, colour, 2)

            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colour, 2)
            cv2.putText(annotated, label, (bbox[0], bbox[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2)
            
            current_faces.append({
                "id": display_id,
                "status": "Recognized" if fid else f"Analyzing ({val_count}/5)",
                "color": hex_color,
                "sim": f"{sim:.2f}" if fid else "--"
            })

        # Get Events — last 12, newest first, with formatted time
        recent_events = db.get_latest_events(12)
        formatted_events = []
        for ev in recent_events:
            raw_ts = ev.get("timestamp") or ""
            try:
                # SQLite default format: "2026-03-22 21:10:09"
                dt = datetime.datetime.strptime(raw_ts[:19], "%Y-%m-%d %H:%M:%S")
                friendly_ts = dt.strftime("%H:%M:%S")
            except Exception:
                friendly_ts = raw_ts[-8:] if len(raw_ts) >= 8 else "--"
            # Fix image path: ev['image_path'] already starts with 'logs/...',
            # do not prepend /logs/ again. Convert Windows separators for browser.
            img_path = ev.get("image_path")
            img_url = ("/" + img_path.replace("\\", "/")) if img_path else None
            formatted_events.append({
                "type": ev["event_type"],
                "face_id": ev["face_id"],
                "time": friendly_ts,
                "image": img_url
            })

        with lock:
            # We encode to JPEG here to avoid heavy encoding inside the Flask route loop
            ret, buffer = cv2.imencode('.jpg', annotated)
            if ret:
                locked_frame = buffer.tobytes()  # type: ignore[union-attr]
            locked_stats["fps"] = round(fps_val)
            locked_stats["global_count"] = global_count
            locked_stats["session_count"] = len(session_unique_ids)
            locked_stats["faces"] = current_faces
            locked_stats["events"] = formatted_events
            locked_stats["latency_ms"] = int(last_inference_ms)

@app.route("/")
def index():
    return render_template("index.html")

def gen_frames():
    while True:
        with lock:
            frame_bytes = locked_frame
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/stats")
def stats():
    with lock:
        st = copy.deepcopy(locked_stats)
    return jsonify(st)

@app.route("/api/videos")
def get_videos():
    import glob
    videos = []
    if os.path.exists("sample"):
        videos = [os.path.basename(p) for p in glob.glob("sample/*.mp4")]
    # sort numerically if possible properly mapped to tuples to avoid TypeError on mixed arrays
    videos.sort(key=lambda x: (0, int(x.split('.')[0])) if x.split('.')[0].isdigit() else (1, x))
    return jsonify({"videos": videos})

@app.route("/api/set_video", methods=["POST"])
def set_video():
    global active_source, trigger_reset
    from flask import request  # type: ignore
    data = request.json or {}
    vid = data.get("video")
    if not vid:
        return jsonify({"error": "No video provided"}), 400
    
    path_to_vid = os.path.join("sample", vid)
    if not os.path.exists(path_to_vid) and vid != "0":
        return jsonify({"error": "Video not found"}), 404
        
    with config_lock:
        active_source = path_to_vid if vid != "0" else 0
        trigger_reset = True
        
    return jsonify({"status": "success", "video": vid})

@app.route("/logs/<path:filepath>")
def serve_logs(filepath):
    # Ensure filepath doesn't accidentally escape `logs/`
    target_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(target_dir, filepath)

if __name__ == "__main__":
    t = threading.Thread(target=tracker_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
