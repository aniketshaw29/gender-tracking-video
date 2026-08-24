# Centroid Tracking

The centroid tracker (`tracker.py`) handles **frame-to-frame identity persistence**: given new face detections each frame, which detection is the same person as last frame?

Without a tracker, every frame would be a fresh set of anonymous faces with no connection to the previous frame — the same person would get a new random ID every time.

---

## The core idea: centroids

A **centroid** is the centre point of a bounding box:

```
box: (x, y, w, h)
centroid: (x + w/2,  y + h/2)
```

Centroids reduce a face rectangle to a single 2D point. Matching identities across frames becomes a point-matching problem.

---

## Algorithm

### Data structures

```python
objects:    {id → centroid}   # currently tracked faces
disappeared: {id → int}       # consecutive missed frames per ID
```

### On each `update(boxes)` call

**No detections this frame** — increment every `disappeared` counter. Deregister any ID that exceeds `max_disappeared` (default 60, ~2 s at 30 fps).

**No existing objects** — register every detection as a fresh ID.

**Normal case (both existing objects and new detections)**

1. Compute a pairwise Euclidean distance matrix `D[i][j]` between every existing centroid `i` and every new detection centroid `j`.

2. Greedy nearest-neighbour matching:
   - Sort existing objects by their nearest detection distance (smallest first)
   - For each (object, detection) pair, if neither has been matched yet, pair them: update the object's centroid and reset its `disappeared` to 0

3. Unmatched existing objects: increment `disappeared`; deregister if over threshold.

4. Unmatched detections: register as new IDs (new people entered frame).

### Why greedy instead of the Hungarian algorithm?

The Hungarian algorithm finds the globally optimal assignment. Greedy can fail when one short-distance pair "steals" a match that would have been globally better. In practice for face tracking, greedy works because:
- People rarely pass through each other in 2D frame space
- High frame rates mean centroids move small distances between frames
- Simplicity makes it fast and debuggable

For dense scenes (10+ people), consider `scipy.optimize.linear_sum_assignment`.

---

## `max_disappeared`

When a face disappears (person looks away, partial occlusion), the tracker doesn't immediately discard the ID. It waits `max_disappeared` frames before deregistering.

Current default: **60 frames** (~2 s at 30 fps). A person can blink, briefly turn their head, or be partially blocked and still recover their original ID.

| Lower value | Higher value |
|---|---|
| Faster cleanup | Longer memory for occlusions |
| More risk of re-ID on brief disappearances | Handles longer gaps gracefully |

---

## Relationship to PersonDatabase

The centroid tracker assigns `obj_id`s — these are **short-lived**. Once a person leaves and the tracker deregisters them, the `obj_id` is gone.

The `PersonDatabase` (`reid.py`) is the **long-lived** layer on top: it maps each `obj_id` to a `person_id` that persists for the entire session. If the same person re-enters the frame 5 minutes later, the centroid tracker gives them a new `obj_id`, but `PersonDatabase` recognises their position and maps them back to their original `person_id`.

```
obj_id (tracker)    →    person_id (PersonDatabase)
  0                 →        0         ← first visit
  1   (same person) →        0         ← returned after leaving
  2   (new person)  →        1         ← genuinely new person
```

See [person-reidentification.md](person-reidentification.md) for how `PersonDatabase` works.

---

## `nearest_box` in `main.py`

The tracker stores centroids; the gender classifier and drawing code need the full bounding box. `nearest_box()` finds the face box whose centre is closest to a given centroid:

```python
def nearest_box(centroid, faces, max_dist=80):
    # returns None if closest box centre is > 80 px away
```

`max_dist=80` prevents associating a tracked centroid with a distant face box — which can happen when the tracker keeps an ID alive for a disappeared face and a different person is detected nearby.
