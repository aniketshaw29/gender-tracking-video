import threading
import numpy as np


class PersonDatabase:
    """
    Persistent identity store for the session.
    Each unique physical person has one entry that survives across centroid tracker resets.
    Matching is by cosine similarity of Facenet face embeddings.
    """

    def __init__(self, similarity_threshold: float = 0.60) -> None:
        self.threshold = similarity_threshold
        self._lock = threading.Lock()
        self._next_id = 0
        self._persons: list[dict] = []  # {id, embedding, gender}

    def identify(self, embedding: np.ndarray) -> tuple[int, str | None, bool]:
        """
        Match embedding against all known persons.
        Returns (person_id, gender, is_new).
        is_new=True means this physical person has never been seen this session.
        On match, updates the stored embedding with a running average for robustness.
        """
        with self._lock:
            best_sim, best_person = 0.0, None
            for person in self._persons:
                sim = _cosine(embedding, person["embedding"])
                if sim > best_sim:
                    best_sim, best_person = sim, person

            if best_sim >= self.threshold and best_person is not None:
                # running average keeps the embedding current without drifting too fast
                best_person["embedding"] = 0.8 * best_person["embedding"] + 0.2 * embedding
                return best_person["id"], best_person["gender"], False

            person_id = self._next_id
            self._next_id += 1
            self._persons.append({
                "id": person_id,
                "embedding": embedding.copy(),
                "gender": None,
            })
            return person_id, None, True

    def set_gender(self, person_id: int, gender: str) -> None:
        with self._lock:
            for person in self._persons:
                if person["id"] == person_id:
                    person["gender"] = gender
                    return

    def get_gender(self, person_id: int) -> str | None:
        with self._lock:
            for person in self._persons:
                if person["id"] == person_id:
                    return person["gender"]
        return None


def embed(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
    """Compute a Facenet face embedding for the face at `box` in `frame`."""
    x, y, w, h = box
    roi = frame[y : y + h, x : x + w]
    if roi.size == 0:
        return None
    try:
        from deepface import DeepFace
        result = DeepFace.represent(
            roi,
            model_name="Facenet",
            enforce_detection=False,
            silent=True,
        )
        return np.array(result[0]["embedding"])
    except Exception:
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
