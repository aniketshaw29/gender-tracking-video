# gender-tracking-video

Real-time gender detection and tracking from a webcam using Python, OpenCV, and DeepFace.

Each person that enters the frame is assigned a stable ID and classified as Man/Woman. A running total is displayed on screen and printed at the end of the session.

## How it works

```
Camera → Face Detection → Centroid Tracker (stable IDs) → Gender Classification → Display
```

| Component | File | What it does |
|---|---|---|
| Face detection | [detector.py](detector.py) | OpenCV DNN (ResNet SSD) or Haar cascade fallback |
| Gender classification | [classifier.py](classifier.py) | DeepFace `analyze(actions=["gender"])` |
| Person tracking | [tracker.py](tracker.py) | Centroid tracker — matches faces across frames by nearest distance |
| Main loop | [main.py](main.py) | Camera capture, orchestration, display |

## Setup

```bash
pip install -r requirements.txt

# Download the better DNN face detector (optional but recommended)
python download_models.py
```

## Run

```bash
python main.py                  # default camera (index 0)
python main.py --camera 1       # use a different camera
python main.py --classify-every 20  # re-classify every 20 frames instead of 15
```

Press **Q** to quit. Session totals are printed to the terminal on exit.

## Controls

| Key | Action |
|---|---|
| Q | Quit and print totals |

## Notes

- DeepFace downloads its own model weights on first run (~100 MB).
- Gender classification is not re-run every frame — only every `--classify-every` frames per tracked ID — to keep the loop responsive.
- The DNN face detector is more accurate than the Haar fallback, especially at angles and in poor lighting.
