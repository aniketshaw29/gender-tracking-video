from collections import OrderedDict
import numpy as np


class CentroidTracker:
    """Assigns stable integer IDs to detected faces across video frames."""

    def __init__(self, max_disappeared: int = 50) -> None:
        self.next_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()
        self.disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.just_deregistered: list[int] = []  # IDs removed this update call

    def update(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> OrderedDict[int, np.ndarray]:
        self.just_deregistered = []

        if not boxes:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return self.objects

        centroids = np.array(
            [(x + w // 2, y + h // 2) for x, y, w, h in boxes], dtype=float
        )

        if not self.objects:
            for c in centroids:
                self._register(c)
            return self.objects

        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))

        # pairwise Euclidean distances: rows = existing objects, cols = new detections
        D = np.linalg.norm(obj_centroids[:, None] - centroids[None, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            obj_id = obj_ids[row]
            self.objects[obj_id] = centroids[col]
            self.disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        for row in set(range(len(obj_ids))) - used_rows:
            obj_id = obj_ids[row]
            self.disappeared[obj_id] += 1
            if self.disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        for col in set(range(len(centroids))) - used_cols:
            self._register(centroids[col])

        return self.objects

    def _register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, obj_id: int) -> None:
        self.just_deregistered.append(obj_id)
        del self.objects[obj_id]
        del self.disappeared[obj_id]

