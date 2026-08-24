# Architecture

## Overview

The pipeline is a linear sequence of four stages that runs on every video frame:

```
┌─────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐    ┌─────────┐
│  Camera │ →  │  Face Detector  │ →  │ Centroid Tracker│ →  │Gender Classifier │ →  │ Display │
└─────────┘    └─────────────────┘    └─────────────────┘    └──────────────────┘    └─────────┘
  cv2.Video      detector.py             tracker.py             classifier.py          cv2.imshow
  Capture(0)    Returns bounding        Assigns stable          Calls DeepFace         Draws boxes,
                boxes: [(x,y,w,h)]      integer IDs             every N frames         labels, totals
```

---

## Components

### Camera (`main.py`)

`cv2.VideoCapture(camera_index)` opens the camera device. On macOS, index `0` is the built-in FaceTime camera. On Linux, it maps to `/dev/video0`. The main loop calls `cap.read()` on every iteration, which blocks until a frame is available — this is what drives the real-time behaviour.

### Face Detector (`detector.py`)

Takes a BGR frame (NumPy array `H×W×3`) and returns a list of bounding boxes `[(x, y, w, h)]` — top-left corner plus width/height.

Two implementations are available:

| Mode | Model | Notes |
|---|---|---|
| DNN | ResNet-10 SSD (OpenCV) | More accurate, handles angles & partial faces |
| Haar | `haarcascade_frontalface_default.xml` | Built into OpenCV, no download needed, frontal only |

The DNN mode is used automatically if `models/` contains the downloaded weights; otherwise Haar is used as a fallback.

See [face-detection.md](face-detection.md) for a detailed explanation of both approaches.

### Centroid Tracker (`tracker.py`)

Takes the list of bounding boxes from the detector and returns a dictionary mapping integer IDs to centroids: `{0: [320, 240], 1: [150, 180], ...}`.

The tracker's job is **identity persistence** — making sure the same person keeps the same ID across consecutive frames even as they move. Without tracking, every frame would be a fresh set of anonymous faces with no memory of the previous frame.

See [tracking.md](tracking.md) for the algorithm.

### Gender Classifier (`classifier.py`)

Takes the original frame and a bounding box, crops the face region, and passes it to DeepFace's `analyze()` function with `actions=["gender"]`. Returns `"Man"` or `"Woman"`.

Importantly, this is **not called every frame** for every face. The main loop in `main.py` throttles calls to once every `classify_every` frames (default: 15) per tracked ID. The last known gender label is reused in the frames between calls.

See [gender-classification.md](gender-classification.md) for how DeepFace works internally.

### Display (`main.py`)

After all detections, tracks, and classifications are resolved for a frame, OpenCV draws:

- A coloured rectangle around each detected face
- A text label `ID:N  Man/Woman` above the box
- Running totals in the top-left corner

Blue boxes = Woman, orange boxes = Man.

---

## Data flow in detail

```
cap.read()
  │
  ▼
frame: np.ndarray (H×W×3, BGR)
  │
  ▼
FaceDetector.detect(frame)
  │  returns: [(x,y,w,h), ...]
  ▼
CentroidTracker.update(boxes)
  │  computes centroids from boxes
  │  matches to existing tracked objects by nearest distance
  │  returns: {id: centroid_array, ...}
  ▼
for each tracked ID:
  │  find which face box this centroid belongs to (nearest_box())
  │  if frames_since_last_classify >= classify_every:
  │    GenderClassifier.classify(frame, box)
  │      → DeepFace.analyze(roi, actions=["gender"])
  │      → returns "Man" or "Woman"
  │    store result in id_gender[obj_id]
  │    if first time seeing this ID: increment totals
  ▼
draw boxes, labels, totals onto frame
  │
  ▼
cv2.imshow("Gender Tracking", frame)
```

---

## State held across frames

| Variable | Lives in | What it stores |
|---|---|---|
| `tracker.objects` | `CentroidTracker` | `{id: centroid}` for all currently visible people |
| `tracker.disappeared` | `CentroidTracker` | How many consecutive frames each ID has been missing |
| `id_gender` | `main.py` | Last known gender label per tracked ID |
| `id_last_frame` | `main.py` | Frame index when each ID was last classified |
| `seen_ids` | `main.py` | Set of IDs already counted in totals (prevents double-counting) |
| `totals` | `main.py` | Running `{"Man": N, "Woman": N}` count |

---

## Threading model

The current implementation is **single-threaded**. The main loop blocks on `cap.read()` and then runs detection, tracking, and classification sequentially. Classification is the bottleneck — DeepFace runs a neural network inference for each face. The `classify_every` throttle keeps this manageable.

A more advanced setup would run classification in a background thread with a queue, so the display loop is never blocked. This is a natural next step if frame rate becomes an issue.
