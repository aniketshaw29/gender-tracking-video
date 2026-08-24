# Face Capture

When a person's gender is confirmed for the first time, their face is automatically saved as a JPEG to the `captures/` directory in the project root.

---

## When capture happens

Capture is triggered inside the main loop at the moment a new person is counted:

```python
if person_id not in counted_persons:
    counted_persons.add(person_id)
    totals[gender] += 1
    save_face(frame, box, person_id, gender)   # ← capture
```

Each person is captured **exactly once per session** — the same `person_id` check that prevents double-counting also prevents duplicate captures.

---

## Filename format

```
captures/person_000_Man_20260825_143022.jpg
          ─────────────────────────────────
          person_{id:03d}_{gender}_{YYYYMMDD}_{HHMMSS}.jpg
```

Example files after a short session:

```
captures/
├── person_000_Man_20260825_143022.jpg
├── person_001_Woman_20260825_143105.jpg
└── person_002_Man_20260825_143241.jpg
```

---

## Padding

The saved crop includes **20% padding** around the detected face box on each side. This ensures the image includes forehead, chin, and some background — making it easier to visually confirm identity — rather than a tight crop that clips hair or ears.

```python
pad_x = int(w * 0.2)
pad_y = int(h * 0.2)
x1 = max(0, x - pad_x)
y1 = max(0, y - pad_y)
x2 = min(frame.shape[1], x + w + pad_x)
y2 = min(frame.shape[0], y + h + pad_y)
```

The `max(0, ...)` and `min(frame.shape[...], ...)` clamps prevent the crop from going out of frame bounds.

---

## Terminal output

Each saved file prints a confirmation line:

```
[capture] saved captures/person_000_Man_20260825_143022.jpg
```

---

## Storage location

The `captures/` directory is created automatically on first save (`os.makedirs(CAPTURES_DIR, exist_ok=True)`). It is excluded from git via `.gitignore` — captured photos of real people should not be committed to version control.

---

## Clearing captures between sessions

`captures/` is not cleared automatically on startup. To start fresh:

```bash
rm -rf captures/
```

Old images will remain until manually deleted.
