# Gender Classification

Gender classification is handled by [DeepFace](https://github.com/serengil/deepface), a high-level Python library that wraps several state-of-the-art face analysis models.

---

## What DeepFace does

When you call:

```python
result = DeepFace.analyze(roi, actions=["gender"], enforce_detection=False, silent=True)
```

DeepFace:

1. Tries to detect and align a face within the `roi` (region of interest — the cropped face)
2. Passes the aligned face through a **gender classification neural network**
3. Returns a dictionary with a `"gender"` key containing probability scores:

```python
{
    "gender": {
        "Man": 97.3,    # percentage confidence
        "Woman": 2.7,
    },
    "dominant_gender": "Man",
    ...
}
```

We take `max(scores, key=scores.get)` to get the predicted label.

---

## The model behind it

DeepFace's gender classifier is a **convolutional neural network** trained on the [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) dataset — a large-scale face dataset with millions of images of thousands of identities.

The architecture is based on **VGG-Face**, a deep CNN originally designed for face recognition. The gender branch replaces the identity classification head (thousands of classes) with a 2-class softmax head (Man / Woman).

### Model pipeline

```
Face crop (ROI)
    │
    ▼
Resize to 224×224
    │
    ▼
VGG-Face feature extractor
(13 convolutional layers, max-pooling, batch norm)
    │
    ▼
Global average pooling
    │
    ▼
Dense(512) → ReLU → Dropout
    │
    ▼
Dense(2) → Softmax  →  { "Man": p, "Woman": 1-p }
```

### First-run model download

The first time you run the classifier, DeepFace downloads its model weights into `~/.deepface/weights/`. This is about **100 MB** and happens automatically. Subsequent runs load from disk instantly.

---

## `enforce_detection=False`

By default, DeepFace raises an exception if it can't detect a face in the input image. We set `enforce_detection=False` because we've already detected the face with our own detector — the `roi` we pass in is already a cropped face region. If DeepFace's internal detector disagrees (common when the face is small or at the edge of the frame), this flag prevents a crash and lets it do its best with what it has.

---

## `silent=True`

Suppresses DeepFace's progress bar output that would otherwise clutter the terminal on every call.

---

## Why not every frame?

DeepFace runs a full neural network forward pass for each call — typically **50–200 ms** depending on hardware. At 30 fps, each frame is only ~33 ms, so calling classify every frame would make the display lag by multiple seconds.

The solution in `main.py`:

```python
if frame_idx - id_last_frame.get(obj_id, -classify_every) >= classify_every:
    gender = classifier.classify(frame, box)
```

Classification happens at most once every `classify_every` frames (default: 15) per tracked ID. In between calls, the last known label is displayed unchanged. This is almost invisible to the eye — gender doesn't change frame to frame — while keeping the loop running at near-realtime speed.

---

## Limitations to be aware of

- **Binary classification**: DeepFace outputs Man/Woman. There is no "uncertain" or "non-binary" category.
- **Requires a reasonable face crop**: Very small faces (< 30×30 px) or extreme angles lead to wrong or inconsistent predictions.
- **Lighting sensitivity**: Harsh shadows across the face reduce accuracy.
- **Training data bias**: Like all face analysis models, accuracy varies across demographics depending on how the training data was distributed.

---

## Confidence scores

The raw `gender` dict contains two scores that sum to 100. If you want to display confidence or skip low-confidence predictions, you can modify `classifier.py`:

```python
scores: dict[str, float] = result.get("gender", {})
best = max(scores, key=scores.get)
confidence = scores[best]

if confidence < 70:
    return None  # skip uncertain predictions
return best
```
