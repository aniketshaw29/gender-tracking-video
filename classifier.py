import numpy as np


class GenderClassifier:
    def __init__(self) -> None:
        from deepface import DeepFace  # deferred so import errors surface at runtime
        self._analyze = DeepFace.analyze

    def classify(self, frame: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
        x, y, w, h = box
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            return None
        try:
            result = self._analyze(
                roi,
                actions=["gender"],
                enforce_detection=False,
                silent=True,
            )
            if isinstance(result, list):
                result = result[0]
            scores: dict[str, float] = result.get("gender", {})
            return max(scores, key=scores.get) if scores else None
        except Exception:
            return None
