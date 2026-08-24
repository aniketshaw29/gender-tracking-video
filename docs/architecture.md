# Architecture

## Overview

The pipeline has five stages that run on every video frame:

```
┌─────────┐   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌─────────┐
│  Camera │ → │ Face Detector│ → │ Centroid Tracker │ → │ PersonDB     │ → │ Display │
└─────────┘   └──────────────┘   └──────────────────┘   └──────────────┘   └─────────┘
  cv2.Video     detector.py         tracker.py              reid.py          cv2.imshow
  Capture(0)    [(x,y,w,h)]         {id: centroid}          person_id,       Boxes, labels,
                                    stable IDs              gender           totals
                                                    ↑
                                             classifier.py
                                             (background thread,
                                              every N frames)
```

---

## Components

### Camera (`main.py`)

`cv2.VideoCapture(camera_index)` opens the camera. On macOS, index `0` is the built-in FaceTime camera. The main loop calls `cap.read()` on every iteration, which blocks until a new frame is available.

### Face Detector (`detector.py`)

Takes a BGR frame `(H×W×3)` and returns `[(x, y, w, h)]` — bounding boxes of every detected face.

Two modes, chosen automatically:

| Mode | Model | Used when |
|---|---|---|
| DNN | ResNet-10 SSD | `models/` dir has downloaded weights |
| Haar | `haarcascade_frontalface_default.xml` | Fallback, no download needed |

See [face-detection.md](face-detection.md) for details.

### Centroid Tracker (`tracker.py`)

Takes bounding boxes, converts them to centroids `(cx, cy)`, and matches each centroid to the closest one from the previous frame. Returns `{obj_id: centroid}` — a stable integer ID per visible face that persists across frames.

Deregisters an ID after `max_disappeared=60` consecutive frames without a detection (~2 s at 30 fps).

See [tracking.md](tracking.md) for the algorithm.

### Person Database (`reid.py`)

Maps the short-lived `obj_id` from the centroid tracker to a stable **session-level** `person_id`. Uses **centroid position** for identity: if a returning face appears within `position_radius=200` pixels of a previously seen person's last known position, it is treated as the same person.

Entries are **never removed** during a session — someone who walks out and comes back 5 minutes later still gets their original `person_id`.

See [person-reidentification.md](person-reidentification.md) for details.

### Gender Classifier (`classifier.py`)

Takes the frame and a bounding box, crops the face, and runs DeepFace `analyze(actions=["gender"])`. Returns `"Man"` or `"Woman"`.

**Not called every frame.** The main loop calls it immediately when a face has no gender yet, then throttles to once every `classify_every` frames (default 15, ~0.5 s at 30 fps). The last known label is reused in between.

Model weights are loaded in a **background thread** at startup using `DeepFace.build_model("Gender")`. The camera and tracker run immediately; the display shows a loading overlay until models are ready.

See [gender-classification.md](gender-classification.md) for DeepFace internals.

### Face Capture (`main.py → save_face()`)

The first time a person's gender is confirmed, their face crop is saved to `captures/` as a JPEG with 20% padding around the box. Each person is captured exactly once per session.

See [face-capture.md](face-capture.md) for details.

### Display (`main.py`)

After all stages resolve for a frame, OpenCV draws:

- Coloured rectangle around each face (orange = Man, blue = Woman)
- `ID:N  Man/Woman` label above the box
- Running `Man: N  Woman: N` totals in the top-left
- Animated "Please wait / Loading gender models..." overlay while models are loading

---

## Data flow (single frame)

```
cap.read()  →  frame (H×W×3 BGR)
    │
    ▼
FaceDetector.detect(frame)
    │  → [(x,y,w,h), ...]
    ▼
CentroidTracker.update(boxes)
    │  → {obj_id: centroid, ...}
    ▼
for each (obj_id, centroid):
    │
    ├─ if new obj_id:
    │     PersonDatabase.identify(centroid, frame_idx)
    │       → person_id  (matches by position, or creates new)
    │
    ├─ else:
    │     PersonDatabase.update_position(person_id, centroid, frame_idx)
    │
    └─ if classifier_ready and (no gender yet OR throttle due):
          GenderClassifier.classify(frame, box)
            → "Man" / "Woman"
          PersonDatabase.set_gender(person_id, gender)
          if first time for this person_id:
              totals[gender] += 1
              save_face(frame, box, person_id, gender)
    │
    ▼
draw boxes / labels / totals
    │
    ▼
cv2.imshow(frame)
```

---

## State held across frames

| Variable | Where | What it stores |
|---|---|---|
| `tracker.objects` | `CentroidTracker` | `{obj_id: centroid}` for all currently visible faces |
| `tracker.disappeared` | `CentroidTracker` | Consecutive missed frames per `obj_id` |
| `person_db._persons` | `PersonDatabase` | Every person seen this session: `{id, centroid, gender, last_frame}` |
| `centroid_to_person` | `main.py` | `{obj_id: person_id}` — bridge between tracker and PersonDB |
| `counted_persons` | `main.py` | Set of `person_id`s already in the totals |
| `id_last_frame` | `main.py` | Frame index when each `obj_id` was last classified |
| `totals` | `main.py` | `{"Man": N, "Woman": N}` running count |

---

## Threading model

| Thread | What it does |
|---|---|
| Main thread | Camera loop — detect, track, identify, draw, display |
| Background thread | Loads DeepFace gender model weights at startup, then exits |

The background thread calls `DeepFace.build_model("Gender")` to force weights into memory, then sets `classifier_ready` (a `threading.Event`). Until that event is set, the main loop skips classification and shows a loading overlay.

Classification itself always runs on the main thread — there is no concurrent inference. The throttle (`classify_every`) keeps it from blocking the display loop.
