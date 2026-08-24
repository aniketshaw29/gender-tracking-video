import threading
import numpy as np


class PersonDatabase:
    """
    Persistent identity store using face centroid position.
    Remembers where each person was last seen for `memory_frames` frames.
    If a new face appears within `position_radius` pixels of a remembered person, it is
    treated as the same person — they keep their ID and gender without being counted again.
    """

    def __init__(self, position_radius: int = 200, memory_frames: int = 300) -> None:
        self.position_radius = position_radius
        self.memory_frames = memory_frames
        self._lock = threading.Lock()
        self._next_id = 0
        self._persons: list[dict] = []
        # each entry: {id, centroid, gender, last_frame}

    def identify(
        self, centroid: np.ndarray, current_frame: int
    ) -> tuple[int, str | None, bool]:
        """
        Match centroid against recently seen persons.
        Returns (person_id, gender, is_new).
        is_new=True means this physical person has not been seen this session.
        """
        with self._lock:
            best, best_dist = None, float("inf")
            for p in self._persons:
                if current_frame - p["last_frame"] > self.memory_frames:
                    continue  # entry too old
                dist = float(np.linalg.norm(centroid - p["centroid"]))
                if dist < best_dist:
                    best_dist, best = dist, p

            if best is not None and best_dist <= self.position_radius:
                best["centroid"] = centroid.copy()
                best["last_frame"] = current_frame
                return best["id"], best["gender"], False

            person_id = self._next_id
            self._next_id += 1
            self._persons.append({
                "id": person_id,
                "centroid": centroid.copy(),
                "gender": None,
                "last_frame": current_frame,
            })
            return person_id, None, True

    def update_position(self, person_id: int, centroid: np.ndarray, current_frame: int) -> None:
        with self._lock:
            for p in self._persons:
                if p["id"] == person_id:
                    p["centroid"] = centroid.copy()
                    p["last_frame"] = current_frame
                    return

    def set_gender(self, person_id: int, gender: str) -> None:
        with self._lock:
            for p in self._persons:
                if p["id"] == person_id:
                    p["gender"] = gender
                    return

    def get_gender(self, person_id: int) -> str | None:
        with self._lock:
            for p in self._persons:
                if p["id"] == person_id:
                    return p["gender"]
        return None
