# gender-tracking-video

Real-time gender detection and tracking from a webcam using Python, OpenCV, and DeepFace.

Each person that enters the frame is assigned a stable ID and classified as Man/Woman. A running total is displayed on screen and printed at the end of the session.

---

## Quick start

### 1. Clone and enter the project

```bash
git clone https://github.com/aniketshaw29/gender-tracking-video.git
cd gender-tracking-video
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### 3. Install dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

This installs OpenCV, DeepFace, TensorFlow/Keras, and NumPy. Expect ~2–5 minutes.

### 4. Download the face detector model (recommended)

```bash
.venv/bin/python download_models.py
```

This downloads OpenCV's ResNet SSD face detector into `models/`. Without it, the app falls back to a Haar cascade which is less accurate. See [docs/face-detection.md](docs/face-detection.md) for the difference.

### 5. Run

```bash
.venv/bin/python main.py
```

Press **Q** to quit. Session totals are printed to the terminal on exit.

> **macOS note:** Use `.venv/bin/python` explicitly rather than `python`. On macOS the bare `python` command resolves to the system framework Python at `/Library/Frameworks/Python.framework/...`, which ignores the virtual environment and won't find the installed packages.

---

## Options

```bash
.venv/bin/python main.py --camera 1           # use a different camera index
.venv/bin/python main.py --classify-every 20  # re-classify every 20 frames (default 15)
```

---

## Project files

| File / Folder | Purpose |
|---|---|
| [main.py](main.py) | Entry point — camera loop, orchestration, display |
| [detector.py](detector.py) | Face detection (DNN or Haar fallback) |
| [classifier.py](classifier.py) | Gender classification via DeepFace |
| [tracker.py](tracker.py) | Centroid tracker — stable IDs across frames |
| [reid.py](reid.py) | Person re-identification via position memory |
| [download_models.py](download_models.py) | Downloads the ResNet SSD face detector weights |
| [requirements.txt](requirements.txt) | Python dependencies |
| `captures/` | Auto-created at runtime — one JPEG saved per unique person when their gender is first confirmed |

> Captured photos are named `person_000_Man_20260825_143012.jpg` (ID, gender, timestamp). The `captures/` folder is excluded from git via `.gitignore`.

---

## Documentation

| Doc | What it covers |
|---|---|
| [docs/architecture.md](docs/architecture.md) | System overview, data flow, component responsibilities |
| [docs/face-detection.md](docs/face-detection.md) | How DNN and Haar face detection work |
| [docs/gender-classification.md](docs/gender-classification.md) | How DeepFace classifies gender |
| [docs/tracking.md](docs/tracking.md) | Centroid tracking algorithm |
| [docs/performance.md](docs/performance.md) | Tuning for speed and accuracy |

---

## Deactivate the virtual environment

```bash
deactivate
```
