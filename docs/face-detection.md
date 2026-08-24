# Face Detection

Face detection is the first stage in the pipeline. It finds every face in a video frame and returns its location as a bounding box `(x, y, w, h)` — the pixel coordinates of the top-left corner, plus width and height.

This project supports two detection methods and picks between them automatically.

---

## Method 1: OpenCV DNN — ResNet SSD (recommended)

This is used when you run `python download_models.py` first.

### What is it?

A **Single Shot Detector (SSD)** built on top of a **ResNet-10** backbone, trained by OpenCV's team specifically for face detection. The model is tiny — only about 2 MB — but significantly outperforms Haar cascades in practice.

### How it works

1. The input frame is resized to 300×300 pixels and converted into a "blob" — a 4D array `(1, 3, 300, 300)` with pixel values mean-subtracted by `(104, 177, 123)` (the mean RGB values of the training set). Mean subtraction centres the pixel distribution around zero, which helps the network learn faster and generalize better.

2. The blob is passed through the ResNet-10 backbone, which progressively extracts features at multiple scales through convolutional layers.

3. The SSD head runs a sliding detection at multiple aspect ratios and scales simultaneously (unlike older approaches that ran a separate pass for each scale). For each candidate region, it predicts:
   - A confidence score: "how likely is this a face?"
   - Offsets from the anchor box to the actual face location

4. Any detection with `confidence > 0.5` (configurable) is kept and its coordinates are projected back to the original frame size.

### Why it's better than Haar

- Handles **tilted and partially visible** faces that Haar misses entirely
- Works at **lower resolution and in poor lighting**
- Far fewer false positives (random objects detected as faces)
- Runs on OpenCV's DNN module — no separate heavy framework needed

### The model files

| File | Size | What it is |
|---|---|---|
| `deploy.prototxt` | ~3 KB | Network architecture definition (Caffe format) |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~10 MB | Trained weights (1.4M gradient steps) |

These are stored in `models/` and excluded from git (`.gitignore`) because binary weight files don't belong in version control.

---

## Method 2: Haar Cascade (fallback)

Used automatically when the DNN model files are not present.

### What is it?

A **Haar cascade classifier** — the classic face detector from Viola & Jones (2001). It ships built into OpenCV and requires no download.

### How it works

Haar cascades are based on **Haar-like features** — simple rectangular patterns that measure differences in pixel intensity. A feature like:

```
┌───┬───┐
│ + │ - │   "left side darker than right side"
└───┴───┘
```

is computed as the difference between the sum of pixel values in the white rectangle and the black rectangle. These sums are computed efficiently using an **integral image** (a pre-computed lookup table where each cell stores the sum of all pixels above and to the left).

The cascade is a **sequence of classifiers** arranged from fast/weak to slow/strong:

1. Apply a tiny, fast classifier to every possible window position and scale
2. Reject windows that are obviously not faces (most windows)
3. Only pass survivors to the next, slower stage
4. Continue cascading — any window rejected at any stage is immediately discarded

This early-exit structure makes the cascade very fast in practice, because nearly all windows are rejected in the first few stages.

### Parameters used

```python
self._cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,   # how much the image is shrunk at each scale step
    minNeighbors=5,    # how many overlapping detections needed to keep one
    minSize=(40, 40),  # ignore faces smaller than 40×40 pixels
)
```

- `scaleFactor=1.1` means each scan uses an image 10% smaller than the last, covering a range of face sizes. Lower = more thorough but slower.
- `minNeighbors=5` suppresses false positives by requiring multiple overlapping detections. Higher = fewer detections but more reliable.

### Limitations

- Trained mostly on **frontal, upright faces**. Tilted or profile faces are often missed.
- Sensitive to lighting — shadows across the face cause misses.
- More false positives than DNN (patterned backgrounds, hands near faces, etc.)

---

## Choosing between them

The code in `detector.py` picks automatically:

```python
if _PROTO.exists() and _WEIGHTS.exists():
    # DNN mode
else:
    # Haar fallback
```

Run `python download_models.py` once to unlock the DNN detector. The downloaded files are small (~10 MB total) and load instantly.
