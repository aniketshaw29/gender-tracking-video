# Performance and Tuning

## The bottleneck: gender classification

DeepFace's `analyze()` runs a full neural network forward pass — **50–200 ms** per call on CPU. At 30 fps each frame is only ~33 ms, so classifying every face every frame makes the loop lag.

The solution is **temporal throttling** with an immediate-on-first-sight exception:

```python
no_gender_yet = person_db.get_gender(person_id) is None
due = no_gender_yet or (frame_idx - last >= classify_every)
if classifier_ready.is_set() and due:
    gender = classifier.classify(frame, box)
```

- First appearance: classify immediately (no wait)
- After that: at most once every `classify_every` frames (default 15, ~0.5 s at 30 fps)

Between calls the last known label is displayed unchanged — this is invisible to the eye since gender doesn't change frame to frame.

---

## Model loading overhead

The gender model takes ~10 seconds to load on first run (TensorFlow initialisation + weight loading from `~/.deepface/weights/`). This is done in a **background thread** using `DeepFace.build_model("Gender")` so the camera and tracker run immediately. An animated overlay is shown during this period.

After the first session, TensorFlow caches compiled graph artefacts so subsequent startups are slightly faster.

---

## Tuning `--classify-every`

```bash
.venv/bin/python main.py --classify-every 10   # faster label updates, more CPU
.venv/bin/python main.py --classify-every 30   # lower CPU, label may lag slightly
```

| Value | Trade-off |
|---|---|
| 5–10 | Reacts faster when new people enter; higher CPU usage |
| 15 (default) | Good balance on modern hardware |
| 20–40 | Slower label response, lower CPU — good for older machines |

---

## Tuning `max_disappeared` (in `main.py`)

Controls how long the centroid tracker holds an ID for an undetected face.

```python
tracker = CentroidTracker(max_disappeared=60)
```

| Value | Trade-off |
|---|---|
| 20–30 | IDs recycled quickly; more risk of new ID on brief occlusion |
| 60 (default) | ~2 s at 30 fps — handles head turns, brief disappearances |
| 90–120 | Better for slow-moving scenes; IDs stay alive longer |

---

## Tuning `position_radius` (in `main.py`)

Controls how far (in pixels) a returning face can be from their last known position and still be recognised as the same person.

```python
person_db = PersonDatabase(position_radius=200)
```

| Value | Trade-off |
|---|---|
| 100–150 | Stricter matching; less risk of confusing nearby different people |
| 200 (default) | Works for typical seated/desk scenarios |
| 250–350 | Better if people tend to re-enter from a different angle or position |

---

## Face detection confidence (`detector.py`)

```python
FaceDetector(confidence_threshold=0.5)
```

| Higher (0.7–0.9) | Lower (0.3–0.5) |
|---|---|
| Fewer false positives | Detects more faces, including at angles |
| May miss real faces in poor lighting | May trigger on non-face regions |

---

## Haar vs DNN speed

| Method | Typical time | Notes |
|---|---|---|
| Haar cascade | ~5–10 ms | No download; frontal faces only |
| DNN ResNet SSD | ~15–30 ms | More accurate; run `download_models.py` once |

On M-series Macs and modern Intel CPUs, DNN runs comfortably at 30 fps. On older hardware, Haar may be needed to keep up.

---

## Approximate per-frame budget (modern laptop, no GPU)

| Stage | Approx. time |
|---|---|
| `cap.read()` | 5–10 ms |
| Face detection (DNN) | 15–25 ms |
| Centroid tracking | < 1 ms |
| PersonDatabase lookup | < 1 ms |
| Gender classification (amortised over 15 frames) | ~7–14 ms |
| Drawing + `imshow` | 2–5 ms |
| **Total** | **~30–55 ms → ~18–30 fps** |

---

## GPU acceleration

If you have an NVIDIA GPU with CUDA, TensorFlow uses it automatically. Classification time drops from ~150 ms to ~10–20 ms, removing the bottleneck entirely. No code changes needed.

Verify GPU is detected:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Next steps for higher performance

1. **Background classification thread** — put face crops on a queue, classify in a worker thread, write results back. The display loop never blocks on inference.
2. **Batch inference** — classify all faces in a frame in a single `DeepFace.analyze` call instead of one at a time.
3. **Lighter model** — replace DeepFace with a MobileNet-based gender classifier that runs in < 5 ms per face.
4. **Reduce detection resolution** — scale frame down before detection, scale boxes back up. Detection on 320×240 is much faster than on 1920×1080.
