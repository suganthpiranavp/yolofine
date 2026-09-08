# YOLOv8 + ByteTrack + Re-ID Person Detection, Tracking & IN/OUT Counting System

A real-time **Person Detection, Multi-Object Tracking, Persistent Re-ID, and Virtual Line IN/OUT Counting Application** combining custom trained **YOLOv8**, **ByteTrack**, deep MobileNetV3 + spatial color **IdentityManager**, and **PersonCounter**.

---

## 1. Core Architecture & Pipeline

```text
Input (Video / Webcam)
          │
          ▼
   YOLOv8 Detection  (detector.py)
 (Where is a person?)
          │
          ▼
  Person Filtering  (detector.py)
 (Filter class == 'person')
          │
          ▼
       ByteTrack  (tracker.py)
 (Temporary Track IDs: 1, 2, 7...)
          │
          ▼
 IdentityManager / Re-ID  (identity_manager.py)
 (Persistent Person IDs: P001, P002...)
          │
          ▼
    PersonCounter  (counter.py)
 (Evaluates line crossing for P001)
          │
          ▼
 Visualization & Streamlit UI  (pipeline.py & app.py)
 (IN / OUT / CURRENT / UNIQUE / Event Log)
```

### Components Summary:
* **YOLOv8 (`detector.py`)**: Responsible for **Person Detection** — locates objects in each frame and produces bounding boxes, class labels, and detection confidence scores. Filters strictly for `person` class.
* **ByteTrack (`tracker.py`)**: Responsible for **Person Tracking** — assigns temporary Track IDs to detected persons across consecutive frames using Kalman filter motion predictions.
* **IdentityManager (`identity_manager.py`)**: Responsible for **Persistent Person Re-ID** — maps temporary ByteTrack IDs to persistent person IDs (`P001`, `P002`, ...) using deep MobileNetV3 neural embeddings + spatial anatomical color features.
* **PersonCounter (`counter.py`)**: Responsible for **Virtual Line IN/OUT Counting** — detects when a persistent person (`P001`) crosses a virtual counting line, maintaining IN, OUT, and CURRENT occupancy counts.

---

## 2. Person IN/OUT Counting & Hysteresis Logic

### Persistent ID Keying
Counting state is strictly associated with persistent `person_id`s (`P001`, `P002`, ...), NOT temporary ByteTrack track IDs. If a person temporarily disappears and ByteTrack assigns a new temporary ID upon re-entry, `IdentityManager` restores `P001`, and `PersonCounter` seamlessly continues tracking `P001`'s position without double counting or resetting state.

### Dead-Zone Hysteresis Protection
To prevent false count triggers caused by minor bounding-box jitter or standing near the virtual counting line, `PersonCounter` defines a dead-zone margin around `line_y`:
* `SIDE_A` (Top): `center_y < line_y - margin`
* `SIDE_B` (Bottom): `center_y > line_y + margin`
* `DEAD_ZONE`: `line_y - margin <= center_y <= line_y + margin`

Count events are generated **only** when a persistent person's centroid transitions completely from `SIDE_A → SIDE_B` (IN) or `SIDE_B → SIDE_A` (OUT).

---

## 3. Quick Start & How to Run

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run the Web UI
```bash
python -m streamlit run app.py
```

---

## 4. Key Code Locations (Quick Reference)

| Functionality | File | Class / Function | Description |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Inference** | [`detector.py`](file:///d:/yolov8/detector.py) | `PersonDetector.detect_persons()` | `self.model(frame, conf=conf_threshold)` |
| **ByteTrack Update** | [`tracker.py`](file:///d:/yolov8/tracker.py) | `PersonTracker.track_persons()` | `self.tracker.update(boxes_cpu, frame)` |
| **Persistent Re-ID** | [`identity_manager.py`](file:///d:/yolov8/identity_manager.py) | `IdentityManager.update()` | Maps ByteTrack ID -> `P001`, `P002` |
| **IN/OUT Counting** | [`counter.py`](file:///d:/yolov8/counter.py) | `PersonCounter.update()` | Evaluates line crossing for `P001` |
| **Pipeline Integration**| [`pipeline.py`](file:///d:/yolov8/pipeline.py) | `process_frame()` | Integrates detection, tracking, Re-ID & counter |
| **Streamlit Dashboard** | [`app.py`](file:///d:/yolov8/app.py) | `main()` | Live video playback, control panel & event log |

---

## 5. Verification & Unit Tests

Run unit tests verifying all 9 counting scenarios:
```bash
python tests/test_counter.py
```

### Verified Test Scenarios:
1. **Single Entry**: Person moves `SIDE_A → SIDE_B` (`IN = 1, OUT = 0, CURRENT = 1`).
2. **Stationary Person**: Remains inside for 100+ frames (`IN = 1, OUT = 0, CURRENT = 1`, no frame-by-frame increments).
3. **Dead-Zone Jitter**: Bounding box motion near line inside dead zone generates 0 false counts.
4. **Exit Crossing**: Person moves `SIDE_B → SIDE_A` (`IN = 1, OUT = 1, CURRENT = 0`).
5. **Re-Entry**: Person enters again after exiting (`IN = 2, OUT = 1, CURRENT = 1`).
6. **Track Loss & Re-ID**: ByteTrack ID changes, Re-ID restores `P001`, counting state remains tied to `P001`.
7. **Multiple Identities**: `P001` and `P002` cross independently, generating distinct events.
8. **Simultaneous Crossings**: Multiple persons crossing in the same frame generate individual events.
