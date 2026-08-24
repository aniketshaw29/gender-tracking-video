# Person Re-Identification

Re-identification (re-ID) solves the problem of recognising the **same physical person across multiple visits** within a session — when they leave the camera frame and come back later.

The centroid tracker (`tracker.py`) handles frame-to-frame continuity but forgets a person once they've been gone for `max_disappeared` frames (~2 s). Without re-ID, every re-entry would create a new ID and increment the gender counter again — the same person sitting in front of the camera for an hour would be counted dozens of times.

---

## Approach: position-based identity

`PersonDatabase` in `reid.py` maintains a **session-level store** of every person seen, keyed by their last known centroid position.

When a new centroid appears, `PersonDatabase.identify()` searches all previously seen persons. If any stored centroid is within `position_radius` pixels (default: **200 px**), the new centroid is treated as the same person — they get their original `person_id` back.

```
New centroid at (380, 260)
    │
    ▼
Search PersonDatabase:
    person_id=0  last at (320, 240)  →  distance = 68 px  ← within 200 px → MATCH
    person_id=1  last at (620, 310)  →  distance = 251 px
    │
    ▼
Returns (person_id=0, gender="Man", is_new=False)
```

---

## Why position rather than face embeddings?

An earlier version used **Facenet face embeddings** (cosine similarity of 128-dimensional vectors). This approach was dropped because:

- Facenet downloads ~90 MB of weights on first run
- Cosine similarity between two crops of the same face varies widely with lighting, angle, and face size
- Threshold tuning was fragile — too strict caused new IDs on every re-entry; too loose caused different people to merge
- Any SSL or download failure silently disabled re-ID entirely

Position-based re-ID is:
- **Immediate** — no model load, works from frame 1
- **Reliable** — centroid position is stable and doesn't vary with lighting or angle
- **Sufficient** — for the typical use case (person seated in front of camera), re-entry is almost always to roughly the same spot

---

## Session-level memory

`PersonDatabase` entries are **never removed**. `memory_frames=None` (the default) means the full position history survives for the entire session.

This matters because: if `memory_frames` had a finite value (say 300 frames = 10 s), someone who steps away for 15 seconds would be assigned a new `person_id` on return and counted again. Infinite memory prevents this entirely.

---

## `identify()` flow

```python
def identify(centroid, current_frame) -> (person_id, gender, is_new):
    for each stored person:
        skip if entry is too old (only when memory_frames is set)
        compute distance to their stored centroid
    
    if closest distance <= position_radius:
        update their stored centroid to the new position
        update their last_frame
        return (existing_person_id, stored_gender, is_new=False)
    else:
        create new entry with next available ID
        return (new_person_id, gender=None, is_new=True)
```

---

## Parameters

| Parameter | Default | Effect |
|---|---|---|
| `position_radius` | 200 px | Max distance for a centroid to match an existing person. Increase if people tend to re-enter the frame far from their last position. |
| `memory_frames` | `None` | `None` = remember everyone forever. Set to an integer to expire old entries after N frames. |

### Choosing `position_radius`

At a typical webcam distance (~60–80 cm), a person's face centroid occupies a region roughly 100–200 px wide. A `position_radius` of 200 px covers approximately:

- Leaning forward/backward: usually stays within 50–80 px
- Sitting back down in the same chair: within 100 px
- Walking out and re-entering from the same direction: within 150–200 px

If your camera has a wide angle (captures more of the room), consider increasing to 300 px.

---

## Limitation: nearby different people

If two different people both sit in roughly the same spot at different times, `PersonDatabase` will assign them the same `person_id` and only count them once. This is by design — the system prioritises **not double-counting** over perfect identity separation. In a typical scenario (assigned seating, desk cameras), this is rarely a problem.

---

## How it connects to counting

The main loop in `main.py` maintains a `counted_persons` set of `person_id`s already included in the totals. A person is added to the totals exactly once — the first time their gender is classified. On all subsequent appearances (same `person_id` returned by `identify()`), the `if person_id not in counted_persons` check prevents re-counting.
