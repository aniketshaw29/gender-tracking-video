# Performance and Tuning

## The bottleneck: gender classification

DeepFace's `analyze()` runs a full neural network forward pass, which takes **50–200 ms** per call on CPU. At 30 fps, you have ~33 ms per frame. Calling the classifier on every face every frame would immediately make the loop lag.

The solution is **temporal throttling** — calling the classifier infrequently and reusing the last known label in between.

```python
# in main.py
if frame_idx - id_last_frame.get(obj_id, -classify_every) >= classify_every:
    gender = classifier.classify(frame, box)
```

With `classify_every=15` (the default), a face is re-classified at most once every 15 frames — roughly every 0.5 seconds at 30 fps. The display still updates every frame with the last known label, so there's no visible flicker.

---

## Tuning `--classify-every`

```bash
python main.py --classify-every 10   # faster response, more CPU
python main.py --classify-every 30   # slower response, less CPU
```

| Value | Trade-off |
|---|---|
| Low (5–10) | More accurate (reacts faster to multiple people), higher CPU usage |
| High (20–40) | Lower CPU usage, label may lag briefly when a new person enters |
| Default (15) | Good balance for a single webcam on modern hardware |

---

## Tuning `max_disappeared` (in `tracker.py`)

Controls how long the tracker holds onto an ID for an undetected face.

```python
tracker = CentroidTracker(max_disappeared=30)
```

| Value | Trade-off |
|---|---|
| Low (10–20) | IDs recycled faster, less memory, more chance of re-counting brief occlusions |
| High (50–80) | Better occlusion handling, IDs live longer, more memory used |
| Default (30) | ~1 second at 30 fps — enough for a brief head turn |

---

## Tuning face detection confidence (`detector.py`)

```python
FaceDetector(confidence_threshold=0.5)
```

| Higher threshold (0.7–0.9) | Lower threshold (0.3–0.5) |
|---|---|
| Fewer false positives | Detects more faces |
| May miss real faces in poor lighting | May flag non-face regions |

---

## Haar vs DNN speed

The Haar cascade is **faster** than the DNN detector on CPU — it's a simple comparison tree, not a neural network. On an M-series Mac or a modern Intel CPU, the DNN runs comfortably at 30 fps; on older hardware or a Raspberry Pi, you may want to stick with Haar.

```
Haar cascade:       ~5–10 ms per frame
DNN (ResNet SSD):  ~15–30 ms per frame
```

---

## Approximate frame-rate budget (modern laptop, no GPU)

| Stage | Time |
|---|---|
| `cap.read()` (camera I/O) | ~5–10 ms |
| Face detection (DNN) | ~15–25 ms |
| Centroid tracking | < 1 ms |
| Gender classify (DeepFace, every 15th frame) | ~100–200 ms amortized to ~7–14 ms per frame |
| Drawing + `imshow` | ~2–5 ms |
| **Total** | **~30–55 ms → ~18–30 fps** |

---

## GPU acceleration

If you have an NVIDIA GPU with CUDA, TensorFlow (used by DeepFace) will use it automatically. The classification time drops from ~150 ms to ~10–20 ms, removing the bottleneck entirely. No code changes needed — just ensure CUDA and the correct TensorFlow-GPU build are installed.

To verify GPU is being used:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Next steps for higher performance

1. **Run classifier in a background thread** — put face crops on a queue, have a separate thread classify them and write results back. The display loop never blocks.

2. **Batch multiple faces** — instead of calling `DeepFace.analyze` one face at a time, batch crop all faces in a frame and run one inference pass.

3. **Lighter model** — replace DeepFace with a dedicated gender classifier like [InsightFace](https://github.com/deepinsight/insightface) or a MobileNet-based model that runs in < 5 ms.

4. **Reduce detection resolution** — scale the frame down before detection, then scale bounding boxes back up. Detection on a 320×240 frame is much faster than on a 1920×1080 frame.
