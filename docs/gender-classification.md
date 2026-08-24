# Gender Classification

Gender classification is handled by [DeepFace](https://github.com/serengil/deepface), a Python library that wraps several state-of-the-art face analysis models.

---

## What happens on each call

```python
result = DeepFace.analyze(roi, actions=["gender"], enforce_detection=False, silent=True)
```

1. DeepFace tries to align the face within the `roi` (already a cropped face region)
2. The aligned face is passed through the gender neural network
3. Returns probability scores:

```python
{
    "gender": {"Man": 97.3, "Woman": 2.7},
    "dominant_gender": "Man",
    ...
}
```

`classifier.py` takes `max(scores, key=scores.get)` to get the predicted label — `"Man"` or `"Woman"`.

---

## The model

DeepFace's gender classifier is a **convolutional neural network** trained on [VGGFace2](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/). The architecture is based on VGG-Face — a deep CNN for face recognition — with the identity head replaced by a 2-class softmax.

```
Face crop (ROI)
    │
    ▼
Resize to 224×224
    │
    ▼
VGG-Face feature extractor
(13 conv layers, max-pooling, batch norm)
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

Model weights (~100 MB) are downloaded to `~/.deepface/weights/` on first run by DeepFace automatically. Subsequent runs load from disk.

---

## Model preloading

At startup, the background thread calls:

```python
DeepFace.build_model("Gender")
```

This forces the weights into memory **before** `classifier_ready` is set. Without this, the first real classify call on the main thread would trigger a lazy weight load — adding 3–4 s of lag after the loading banner disappears.

Once `classifier_ready` fires, the first classify call for any face uses pre-loaded weights and returns in ~50–200 ms.

---

## Throttling

DeepFace runs a full neural network forward pass per call — typically 50–200 ms on CPU. Running it every frame would block the display loop.

The main loop throttles classification to avoid this:

```python
no_gender_yet = person_db.get_gender(person_id) is None
due = no_gender_yet or (frame_idx - last >= classify_every)
if classifier_ready.is_set() and due:
    gender = classifier.classify(frame, box)
```

- **First appearance**: `no_gender_yet=True` → classify immediately (no wait)
- **Subsequent frames**: classify once every `classify_every` frames (default 15, ~0.5 s at 30 fps)

The last known label is displayed unchanged between calls — gender doesn't change frame-to-frame so this is invisible to the eye.

---

## `enforce_detection=False`

By default, DeepFace raises an exception if it can't detect a face inside the input image. We set this to `False` because we've already detected the face — the `roi` passed in is already a cropped face region. When DeepFace's internal detector disagrees (common for small or angled faces), this flag prevents a crash and lets it classify with what it has.

---

## `silent=True`

Suppresses DeepFace's progress bar output from cluttering the terminal on every call.

---

## Limitations

- **Binary only**: outputs Man or Woman — no "uncertain" or non-binary category
- **Small faces**: crops under ~30×30 px produce unreliable predictions
- **Lighting**: harsh directional shadows across the face reduce accuracy
- **Training bias**: accuracy varies across demographics based on training data distribution

---

## Confidence filtering (optional)

To skip low-confidence predictions, modify `classifier.py`:

```python
scores = result.get("gender", {})
best = max(scores, key=scores.get)
if scores[best] < 70:
    return None   # skip — main loop will retry next classify_every frames
return best
```
