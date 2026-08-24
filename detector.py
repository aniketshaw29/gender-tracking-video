import cv2
import numpy as np
from pathlib import Path

_MODEL_DIR = Path(__file__).parent / "models"
_PROTO = _MODEL_DIR / "deploy.prototxt"
_WEIGHTS = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
_HAAR = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


class FaceDetector:
    def __init__(self, confidence_threshold: float = 0.5) -> None:
        self.threshold = confidence_threshold
        if _PROTO.exists() and _WEIGHTS.exists():
            self._net = cv2.dnn.readNetFromCaffe(str(_PROTO), str(_WEIGHTS))
            self._mode = "dnn"
        else:
            self._cascade = cv2.CascadeClassifier(_HAAR)
            self._mode = "haar"
            print(
                "[detector] DNN model not found — using Haar cascade fallback.\n"
                f"           Run `python download_models.py` to download better models to {_MODEL_DIR}"
            )

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self._mode == "dnn":
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    def _detect_dnn(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104, 177, 123)
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        boxes: list[tuple[int, int, int, int]] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > self.threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes

    def _detect_haar(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        return [(x, y, w, h) for x, y, w, h in faces] if len(faces) > 0 else []
