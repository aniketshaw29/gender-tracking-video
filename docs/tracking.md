# Centroid Tracking

The centroid tracker (`tracker.py`) solves the **identity persistence** problem: given a new set of detected faces on each frame, which detection corresponds to which person from the previous frame?

Without a tracker, every frame would be a fresh set of anonymous faces. There would be no way to:
- Avoid counting the same person multiple times
- Display a stable label that follows a specific person as they move
- Know when someone new enters the frame vs when an existing person moves

---

## The core idea: centroids

A **centroid** is the centre point of a bounding box:

```
box: (x, y, w, h)
centroid: (x + w/2,  y + h/2)
```

Centroids are easier to work with than full bounding boxes because they reduce a rectangle to a single 2D point. Matching two frames' worth of people is then a problem of matching two sets of points.

---

## Algorithm: frame by frame

### Setup

The tracker maintains two dictionaries:

- `objects: {id → centroid}` — all currently tracked people and their last known centroid
- `disappeared: {id → int}` — how many consecutive frames each person has been missing

### On each call to `update(boxes)`

**Case 1 — No detections in this frame**

Increment every `disappeared` counter by 1. If any ID has been missing for more than `max_disappeared` frames (default: 50), deregister it (remove from both dicts). This cleans up people who have left the frame.

**Case 2 — No existing tracked objects**

Register every new detection as a fresh ID. This handles the very first frame and re-initialisation after all objects disappear.

**Case 3 — Both existing objects and new detections exist (the normal case)**

This is where the matching happens. The goal is to pair each existing tracked object with the detection that most likely represents the same person.

1. **Compute a pairwise distance matrix D**

   For `N` existing objects and `M` new detections:
   ```
   D[i][j] = Euclidean distance from object i's centroid to detection j's centroid
   ```

   If `N=2` and `M=3`:
   ```
         det0   det1   det2
   obj0 [  12,   180,   230 ]
   obj1 [ 220,    15,    95 ]
   ```

2. **Greedy nearest-neighbour matching**

   ```python
   rows = D.min(axis=1).argsort()   # sort existing objects by their nearest detection
   cols = D.argmin(axis=1)[rows]    # for each object, its nearest detection
   ```

   Then iterate through `(row, col)` pairs. If neither has been matched yet, pair them: update object `row`'s centroid to detection `col`'s centroid and reset its `disappeared` counter to 0. Mark both as used.

3. **Handle unmatched existing objects**

   Objects with no match: increment their `disappeared` counter. Deregister if it exceeds `max_disappeared`.

4. **Handle unmatched detections**

   Detections with no match: register as new IDs (new people have entered the frame).

### Why greedy and not the Hungarian algorithm?

The Hungarian algorithm finds the **globally optimal** minimum-cost assignment. The greedy approach (sort by smallest distance, match greedily) can fail in edge cases where one pair's short distance "steals" a match from another pair that would have been better overall.

In practice for real-time face tracking, greedy works well because:
- People rarely pass through each other (in 2D frame space)
- Frame rates are high enough that centroids don't jump large distances between frames
- The simplicity makes it fast and debuggable

For more demanding scenarios (many people, occlusions), consider upgrading to the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) via `scipy.optimize.linear_sum_assignment`.

---

## `max_disappeared` parameter

When a face disappears (person looks away, moves out of frame, gets occluded), the tracker doesn't immediately discard the ID. It waits `max_disappeared` frames before deregistering.

This prevents a common failure mode: a person briefly looks away for 1–2 frames (blinking, turning slightly), the detector misses them, the tracker would create a new ID when they reappear. With `max_disappeared=30`, they can be undetected for up to 30 frames (~1 second at 30 fps) and still get their original ID back when they reappear.

**Trade-off:**

| Lower `max_disappeared` | Higher `max_disappeared` |
|---|---|
| Faster cleanup of gone persons | Longer memory of gone persons |
| More likely to create new IDs on brief occlusions | Less likely to create spurious new IDs |
| Better for crowded scenes | Better for scenes with occlusion or brief disappearances |

---

## Nearest-box lookup (`nearest_box` in `main.py`)

The tracker tracks centroids, but the gender classifier and the drawing code need the full bounding box. `nearest_box()` bridges this gap:

```python
def nearest_box(centroid, faces, max_dist=80):
    # find the face box whose centre is closest to this centroid
    # return None if the closest is farther than max_dist pixels
```

`max_dist=80` prevents a centroid from being associated with a face box that's far away — which could happen when a tracked person's centroid is still on screen (within `max_disappeared` frames) but no face is detected near them.

---

## Counting unique people

The main loop uses a `seen_ids` set to count each ID only once:

```python
if obj_id not in seen_ids:
    seen_ids.add(obj_id)
    totals[gender] += 1
```

This means:
- A person walking through frame once = counted once
- A person walking out and coming back = counted twice (gets a new ID after `max_disappeared` frames)

If you want to count unique individuals across re-entries, you would need a more sophisticated re-identification step (e.g., comparing face embeddings). The current tracker has no memory of deregistered IDs.
