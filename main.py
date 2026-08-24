import argparse
import cv2
import numpy as np

from detector import FaceDetector
from classifier import GenderClassifier
from tracker import CentroidTracker


def nearest_box(
    centroid: np.ndarray,
    faces: list[tuple[int, int, int, int]],
    max_dist: int = 80,
) -> tuple[int, int, int, int] | None:
    if not faces:
        return None
    best, best_d = None, float("inf")
    for x, y, w, h in faces:
        d = float(np.linalg.norm(centroid - np.array([x + w // 2, y + h // 2])))
        if d < best_d:
            best_d, best = d, (x, y, w, h)
    return best if best_d < max_dist else None


def run(camera_index: int = 0, classify_every: int = 15) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_index}")

    detector = FaceDetector()
    classifier = GenderClassifier()
    tracker = CentroidTracker(max_disappeared=30)

    id_gender: dict[int, str] = {}       # stable gender label per tracked ID
    id_last_frame: dict[int, int] = {}   # frame index of last classification
    seen_ids: set[int] = set()           # IDs already counted in totals
    totals: dict[str, int] = {"Man": 0, "Woman": 0}
    frame_idx = 0

    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        faces = detector.detect(frame)
        tracked = tracker.update(faces)

        for obj_id, centroid in tracked.items():
            box = nearest_box(centroid, faces)
            last = id_last_frame.get(obj_id, -classify_every)
            if box is not None and frame_idx - last >= classify_every:
                gender = classifier.classify(frame, box)
                if gender:
                    id_last_frame[obj_id] = frame_idx
                    id_gender[obj_id] = gender
                    if obj_id not in seen_ids:
                        seen_ids.add(obj_id)
                        totals[gender] = totals.get(gender, 0) + 1

        # --- draw ---
        for obj_id, centroid in tracked.items():
            gender = id_gender.get(obj_id, "?")
            color = (255, 120, 50) if gender == "Man" else (50, 120, 255)
            box = nearest_box(centroid, faces)
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, f"ID:{obj_id}  {gender}",
                    (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
                )

        y_off = 30
        for label, count in totals.items():
            cv2.putText(
                frame, f"{label}: {count}",
                (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2,
            )
            y_off += 32

        cv2.imshow("Gender Tracking  [Q = quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nSession totals:")
    for label, count in totals.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time gender tracking from camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument(
        "--classify-every", type=int, default=15,
        help="Re-classify a tracked face every N frames (default 15)",
    )
    args = parser.parse_args()
    run(args.camera, args.classify_every)
