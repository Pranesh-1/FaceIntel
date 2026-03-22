# Intelligent Face Tracker with Auto-Registration and Visitor Counting

> **This project is a part of a hackathon run by https://katomaran.com**

---

## Overview

An AI-driven, production-grade real-time face tracking and unique visitor counting system built for the **Katomaran Hackathon - March 2026**.

The system processes a live video stream or recorded file to:

- Detect faces using YOLOv8 (person-detection model with ByteTrack)
- Track them persistently across frames using ByteTrack
- Recognise identities via InsightFace (ArcFace) embeddings and cosine similarity
- Auto-register new faces with unique IDs on first detection
- Log every entry and exit event with a timestamped cropped face image
- Count unique visitors accurately across the session without double-counting re-entries

---

## Project Structure

```
FaceIntel/
|-- app.py                      # Flask web server + tracker loop + dashboard API
|-- main.py                     # CLI entry point for headless / RTSP mode
|-- dashboard.py                # Standalone dashboard helper
|-- config.json                 # Configurable runtime parameters
|-- requirements.txt
|-- bytetrack_custom.yaml       # ByteTrack hyperparameters
|
|-- core/
|   |-- tracker.py              # YOLOv8 + ByteTrack wrapper (stable track IDs)
|   |-- embedding.py            # InsightFace ArcFace embedding generator
|   |-- matcher.py              # Cosine similarity face matcher
|   |-- detector.py             # Standalone YOLO face detector (optional)
|
|-- services/
|   |-- face_registry.py        # Central identity hub: registration, re-ID, entry/exit
|   |-- visitor_counter.py      # Unique visitor counter (in-memory + DB-backed)
|   |-- logger.py               # Event logging (events.log + timestamped images)
|
|-- database/
|   |-- db.py                   # SQLite CRUD operations
|   |-- models.py               # Table DDL schema constants
|
|-- utils/
|   |-- config_loader.py        # JSON config reader with typed getters
|   |-- image_utils.py          # Image crop, save, and dated directory helpers
|
|-- templates/
|   |-- index.html              # Real-time web dashboard (Flask + MJPEG stream)
|
|-- logs/
|   |-- events.log              # Rotating system event log (auto-generated)
|   |-- entries/<YYYY-MM-DD>/   # Cropped face images on ENTRY events
|   |-- exits/<YYYY-MM-DD>/     # Cropped face images on EXIT events
|   |-- registereds/<YYYY-MM-DD>/ # Cropped face images on first REGISTRATION
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Pranesh-1/FaceIntel.git
cd FaceIntel
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On first run, `ultralytics` automatically downloads YOLOv8 weights. InsightFace downloads the `buffalo_l` model pack (~200 MB). Internet access is required once.

### 4. Run with a sample video file

```bash
# Run the web dashboard (recommended)
python app.py

# OR headless CLI mode
python main.py --source path/to/video.mp4 --show
```

Then open `http://localhost:5000` in your browser to view the live dashboard.

### 5. Run against an RTSP camera (interview / production mode)

```bash
python main.py --source rtsp://user:password@ip:port/stream
```

Or set `camera_source` in `config.json` to the RTSP URL.

### 6. Reset the database

```bash
python scripts/reset_db.py
```

---

## Sample config.json

```json
{
    "frame_skip": 2,
    "similarity_threshold": 0.45,
    "exit_frame_threshold": 30,
    "camera_source": "0",
    "yolo_model": "yolov8n.pt",
    "db_path": "database/face_tracker.db",
    "log_dir": "logs"
}
```

| Key | Description | Default |
|---|---|---|
| `frame_skip` | Run detection every N frames. Reduces CPU load. | `2` |
| `similarity_threshold` | Minimum cosine similarity to match a known face. | `0.45` |
| `exit_frame_threshold` | Frames a track must be absent before EXIT is logged. | `30` |
| `camera_source` | Video file path, RTSP URL, or `"0"` for webcam. | `"0"` |
| `yolo_model` | YOLO weights file path. | `yolov8n.pt` |
| `db_path` | SQLite database file path. | `database/face_tracker.db` |
| `log_dir` | Root directory for logs and images. | `logs` |

---

## Architecture

```
Video / RTSP Input
        |
        v
  Frame Reader (OpenCV)
        |
        v
  Frame Skipper <-- config.json (frame_skip)
        |
        v
  YOLOv8 + ByteTrack
  (core/tracker.py)         --> stable track_ids per bounding box
        |
        v
  Face Crop (Top 40% of body bbox)
  (utils/image_utils.py)
        |
        v
  InsightFace ArcFace
  (core/embedding.py)       --> 512-d normalised embedding
        |
        v
  Cosine Similarity Matcher
  (core/matcher.py)         --> match_id or None
        |
    +---+-------------------+
    |                       |
  NEW FACE             KNOWN FACE
    |                       |
  Register             Re-identify
  (DB + log)           Log Entry
    |
  Increment VisitorCounter
        |
        v
  Entry / Exit Logger <-- exit_frame_threshold
  (services/logger.py)
        |
  +-----+----------+
  |                |
 DB            Filesystem
(events)    logs/<type>/<date>/
```

### Key Architectural Decision: The Ambiguity Band

ArcFace embeddings on a 512-d hypersphere behave predictably for frontal faces (>0.70 similarity). During development, a non-linear ambiguity band was identified between 0.55 and 0.65.

- **>=0.65**: Strong global match. Same person. Always merged.
- **<0.55**: Definitive mismatch. Always creates a new identity.
- **[0.55, 0.65)**: Safety rescue band — used for temporal re-identification.

The system implements a three-tier flow:

1. **Strict quality gating**: Only faces larger than 30x30 pixels with >0.60 detection confidence are embedded. Eliminates ghost IDs during blurry entries.
2. **Track-body persistence**: Identity mappings are never dropped while a person remains physically tracked by ByteTrack, even with temporary facial occlusion.
3. **Temporal rescue**: If a track is lost and re-acquired within a short window, the ID is rescued if similarity falls in the high-confidence band.

---

## AI Planning Document

### System Design Approach

The pipeline was designed with a production-first mindset following a strict separation of concerns:

| Layer | Responsibility |
|---|---|
| Detection | YOLOv8 identifies persons in the frame |
| Tracking | ByteTrack assigns and maintains stable track IDs across frames |
| Embedding | InsightFace generates discriminative 512-d ArcFace vectors |
| Matching | Cosine similarity maps embeddings to registered identities |
| Registry | FaceRegistry manages identity lifecycle (ENTRY, EXIT, RE-ID) |
| Logger | FaceEventLogger records all events to disk and database |
| Counter | VisitorCounter maintains an accurate unique count |

### AI Tools Used

The following prompts were used during AI-assisted development:

**Master Architecture Prompt**
> "You are a senior AI engineer. Build a production-grade real-time face tracking and recognition system. Requirements: YOLOv8 detection, ByteTrack tracking, InsightFace ArcFace embeddings, cosine similarity matching, auto-registration of new faces with unique IDs, entry/exit event logging with timestamped images, SQLite persistence, configurable frame skipping via config.json. Code must be modular with single-responsibility modules. Avoid the face_recognition library. Suggest a clean folder structure and data flow between components."

**Detection Prompt**
> "Write a Python module for face detection using YOLOv8 (ultralytics). Accept a BGR frame, return xyxy bounding boxes above a confidence threshold. Make it reusable and keep model loading in __init__."

**Tracking Prompt**
> "Implement a face tracking module using ultralytics .track() API with ByteTrack. Return a list of dicts with track_id and bbox per detection. Explain how persistent IDs survive across frames."

**Embedding Prompt**
> "Write a Python module using InsightFace FaceAnalysis with the buffalo_l pack to extract ArcFace embeddings. Input: cropped BGR face image. Output: normalised 512-d numpy array or None if no face detected. Defer import and support CPU/GPU."

**Matching Prompt**
> "Implement cosine similarity face matching. Given a query embedding and a list of (id, embedding) pairs, return (best_id, similarity) or (None, 0.0). Explain why dot product works on L2-normalised vectors."

**Logging Prompt**
> "Design a robust logging system: rotating events.log plus save cropped images in logs/entries/date/ and logs/exits/date/. Each event type: ENTRY, EXIT, REGISTERED, RECOGNISED. Write a FaceEventLogger class that calls a DB instance to persist metadata."

**Exit Detection Prompt**
> "Implement entry/exit detection using a dictionary active_tracks={track_id: last_seen_frame}. An exit is declared when a track_id is absent for more than exit_frame_threshold frames. Ensure exactly one EXIT log per face per appearance."

**Database Prompt**
> "Design a SQLite schema for face tracking with two tables: faces (id, embedding BLOB, created_at) and events (id, face_id FK, event_type, timestamp, image_path). Write Python CRUD using sqlite3 with context managers for safety."

**Performance Optimisation Prompt**
> "The InsightFace ArcFace model is causing frame latency spikes when multiple faces appear simultaneously. Suggest strategies: background daemon threads, per-frame embedding ceilings, embedding cooldowns, and EMA latency smoothing. Implement these without blocking the main tracker loop."

---

## Compute Load Estimation

| Component | CPU Load | GPU Load | Notes |
|---|---|---|---|
| YOLOv8n detection | 20-35% | Low-High (if CUDA) | Optimised for speed |
| ByteTrack | <5% | None | Pure numpy IoU matching |
| InsightFace ArcFace | 40-60% | Medium (if CUDA) | 512-d embedding |
| Cosine matching | Negligible | None | Dot product on small arrays |
| SQLite logging | Negligible | None | Non-blocking small inserts |
| **Total (CPU only)** | **~60-80%** | — | Modern quad-core |
| **Total (GPU)** | **~15-25%** | **~40-60%** | With CUDA |

**Expected throughput:**
- CPU only: 8-15 FPS (1080p input, frame_skip=2)
- GPU (RTX 3060+): 25-35 FPS

---

## Database Schema

**`faces` table** — one row per unique registered person:

| Column | Type | Description |
|---|---|---|
| `id` | TEXT | Unique face ID (`face_xxxxxx`) |
| `embedding` | BLOB | 512-float ArcFace embedding (numpy binary) |
| `created_at` | DATETIME | First registration timestamp |

**`events` table** — one row per entry/exit/recognition event:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `face_id` | TEXT | Foreign key to faces.id |
| `event_type` | TEXT | `ENTRY`, `EXIT`, `REGISTERED`, `RECOGNISED` |
| `timestamp` | DATETIME | Event timestamp |
| `image_path` | TEXT | Relative path to saved crop image |

---

## Sample events.log Output

```
2026-03-23 00:05:01,412 [INFO] services.logger: Starting logging system at logs/
2026-03-23 00:05:03,001 [INFO] services.visitor_counter: Visitor counter initialised at 0 unique visitor(s).
2026-03-23 00:05:04,190 [INFO] services.face_registry: Validation passed for track 1 (count=5, sim=0.00)
2026-03-23 00:05:04,233 [INFO] services.face_registry: NEW FACE registered: track=1 -> face_3a2f1b | Total unique visitors: 1
2026-03-23 00:05:04,234 [INFO] services.logger: [REGISTERED] face_3a2f1b | image=logs/registereds/2026-03-23/face_3a2f1b_00-05-04-234.jpg
2026-03-23 00:05:04,235 [INFO] services.logger: [ENTRY] face_3a2f1b | image=logs/entries/2026-03-23/face_3a2f1b_00-05-04-235.jpg
2026-03-23 00:05:08,310 [INFO] services.face_registry: RECOGNISED: track=1 -> face_3a2f1b (sim=0.712)
2026-03-23 00:05:08,311 [INFO] services.logger: [RECOGNISED] face_3a2f1b
2026-03-23 00:05:11,140 [INFO] services.face_registry: Validation passed for track 3 (count=5, sim=0.00)
2026-03-23 00:05:11,142 [INFO] services.face_registry: NEW FACE registered: track=3 -> face_b7c91a | Total unique visitors: 2
2026-03-23 00:05:11,143 [INFO] services.logger: [REGISTERED] face_b7c91a | image=logs/registereds/2026-03-23/face_b7c91a_00-05-11-143.jpg
2026-03-23 00:05:11,144 [INFO] services.logger: [ENTRY] face_b7c91a | image=logs/entries/2026-03-23/face_b7c91a_00-05-11-144.jpg
2026-03-23 00:05:18,778 [INFO] services.face_registry: EXIT detected: track=1 face=face_3a2f1b at frame=312
2026-03-23 00:05:18,779 [INFO] services.logger: [EXIT] face_3a2f1b | image=logs/exits/2026-03-23/face_3a2f1b_00-05-18-779.jpg
2026-03-23 00:05:20,100 [INFO] services.face_registry: EXIT detected: track=3 face=face_b7c91a at frame=345
2026-03-23 00:05:20,101 [INFO] services.logger: [EXIT] face_b7c91a | image=logs/exits/2026-03-23/face_b7c91a_00-05-20-101.jpg
2026-03-23 00:05:41,100 [INFO] services.face_registry: RECOGNISED: track=7 -> face_3a2f1b (sim=0.731) [re-entry]
2026-03-23 00:05:41,101 [INFO] services.logger: [RECOGNISED] face_3a2f1b
2026-03-23 00:05:41,102 [INFO] services.logger: [ENTRY] face_3a2f1b | image=logs/entries/2026-03-23/face_3a2f1b_00-05-41-102.jpg
2026-03-23 00:06:00,500 [INFO] services.visitor_counter: Synced visitor count from DB: 2
2026-03-23 00:06:00,501 [INFO] main: Session complete | Unique visitors: 2
```

---

## Sample Output Structure

After processing the sample video:

```
logs/
|-- events.log
|-- entries/2026-03-23/
|   |-- face_3a2f1b_00-05-04-235.jpg
|   |-- face_b7c91a_00-05-11-144.jpg
|   |-- face_3a2f1b_00-05-41-102.jpg    (re-entry — not counted again)
|-- exits/2026-03-23/
|   |-- face_3a2f1b_00-05-18-779.jpg
|   |-- face_b7c91a_00-05-20-101.jpg
|-- registereds/2026-03-23/
|   |-- face_3a2f1b_00-05-04-234.jpg
|   |-- face_b7c91a_00-05-11-143.jpg

database/
|-- face_tracker.db   # Queryable with any SQLite browser
```

Unique visitor count retrievable via:

```python
from database.db import Database
db = Database("database/face_tracker.db")
print(db.count_unique_visitors())  # e.g., 2
```

---

## Frame Processing Optimisation

Detection and embedding run only every `frame_skip` frames (configurable via `config.json`). Intermediate frames advance the exit detection counter using stale tracker data, maintaining continuity without redundant inference.

InsightFace inference runs in background daemon threads to prevent blocking the main tracker loop. A per-frame embedding ceiling (max 1 per frame) prevents CPU saturation when many faces appear simultaneously.

---

## Embedding Storage

Facial embeddings are 512-dimensional float32 vectors generated by InsightFace ArcFace. Stored in SQLite as BLOBs:

```python
# Store
embedding.astype(np.float32).tobytes()

# Retrieve
np.frombuffer(blob, dtype=np.float32)
```

---

## Face Matching Strategy

Cosine similarity on L2-normalised vectors reduces to a dot product:

```
similarity = dot(query_embedding, stored_embedding)
```

- `similarity >= threshold` — existing face (re-ID, count unchanged)
- `similarity < threshold` — new face (register, increment count)

The threshold is tunable via `config.json` to handle varying lighting and camera conditions.

---

## Fault Tolerance

| Risk | Strategy |
|---|---|
| DB corruption | All writes use atomic SQLite context managers |
| Log loss | events.log is append-only with rotating backups (10 MB, 5 copies) |
| Image loss | Crops are saved before DB insert confirmation |
| Crash / interrupt | Graceful shutdown flushes all active tracks as EXIT events |

---

## Edge Case Handling

| Scenario | Strategy |
|---|---|
| Temporary occlusion | ByteTrack maintains ID through short gaps |
| Rapid camera re-entry | EXIT triggered only after `exit_frame_threshold` absent frames |
| Multiple similar faces | Controlled via `similarity_threshold` tuning |
| Partial or small faces | Confidence filter + minimum crop-size guard in embedding.py |
| Same person re-entering | Re-identified via embedding match; count not incremented |

---

## Model Selection Justification

| Model | Reason |
|---|---|
| YOLOv8 | State-of-the-art real-time detection, high accuracy, easy ByteTrack integration |
| ByteTrack | Robust multi-object tracking, stable IDs under partial occlusion |
| InsightFace ArcFace | Industry-standard facial embeddings with exceptional discriminative power |
| face_recognition library | Explicitly excluded per hackathon requirements (insufficient production accuracy) |

---

## Scaling Considerations

The system can be extended to:

- Multi-camera setups by running multiple pipeline instances in parallel
- Cloud databases (PostgreSQL, TimescaleDB) by swapping the SQLite layer
- Distributed processing with a message queue (Redis, Kafka)
- GPU-accelerated inference by setting `ctx_id=0` in the InsightFace prepare call

---

## Potential Applications

- Smart surveillance: detect and log persons of interest in real time
- Retail analytics: count footfall and measure dwell time
- Office attendance tracking: automated entry/exit logging without badge scanning
- Secure access monitoring: alert on unregistered individuals entering restricted zones

---

## Assumptions

1. The video source contains frontal or near-frontal faces (within approximately 45 degrees of frontal for reliable ArcFace embeddings).
2. Each unique visitor is defined as a person who appears in the stream; re-entries of the same face are not counted again.
3. Standard `yolov8n.pt` is used for body/person detection; face-specific weights can be substituted as a drop-in replacement.
4. GPU acceleration is optional. The system runs on CPU only with reduced FPS.
5. The similarity threshold of 0.45 is a starting baseline and should be tuned based on lighting conditions and camera angle.
6. The sample videos provided in the Drive link were used for development and testing.

---

## Demo Video

> Watch the Demo: [YouTube / Loom Link to be added]
> *Record and insert link before submission deadline.*

---

## Setup Dependencies Summary

```
ultralytics>=8.0.0
insightface>=0.7.3
opencv-python>=4.8.0
numpy>=1.24.0
flask>=3.0.0
onnxruntime>=1.16.0
```

---

*This project is a part of a hackathon run by https://katomaran.com*
