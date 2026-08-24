import argparse
import threading
import cv2
import numpy as np

from detector import FaceDetector
from classifier import GenderClassifier
from tracker import CentroidTracker
from reid import ReIDStore


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
    tracker = CentroidTracker(max_disappeared=60)
    reid = ReIDStore(similarity_threshold=0.60)

    # Load TensorFlow-backed classifier in background so camera opens immediately
    classifier: GenderClassifier | None = None
    classifier_ready = threading.Event()

    def _load_classifier() -> None:
        nonlocal classifier
        classifier = GenderClassifier()
        classifier_ready.set()

    threading.Thread(target=_load_classifier, daemon=True).start()

    id_gender: dict[int, str] = {}
    id_last_frame: dict[int, int] = {}
    id_embedding: dict[int, np.ndarray] = {}  # face embedding per active ID
    id_from_gallery: set[int] = set()          # IDs re-identified from gallery (don't count)
    seen_ids: set[int] = set()
    totals: dict[str, int] = {"Man": 0, "Woman": 0}
    frame_idx = 0

    print("Press Q to quit.  (gender model loading in background...)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        faces = detector.detect(frame)
        tracked = tracker.update(faces)

        # --- clean up deregistered IDs ---
        for old_id in tracker.just_deregistered:
            if old_id in id_embedding and old_id in id_gender:
                reid.store(id_embedding[old_id], id_gender[old_id])
            id_embedding.pop(old_id, None)
            id_gender.pop(old_id, None)
            id_last_frame.pop(old_id, None)
            id_from_gallery.discard(old_id)

        # --- classify / re-identify ---
        for obj_id, centroid in tracked.items():
            box = nearest_box(centroid, faces)
            if box is None:
                continue

            # new ID: try to match against gallery before classifying
            if obj_id not in id_embedding:
                emb = reid.embed(frame, box)
                if emb is not None:
                    id_embedding[obj_id] = emb
                    gallery_gender = reid.find(emb)
                    if gallery_gender:
                        id_gender[obj_id] = gallery_gender
                        id_from_gallery.add(obj_id)

            # gender classification (skip if already resolved from gallery or model not ready)
            last = id_last_frame.get(obj_id, -classify_every)
            if (
                obj_id not in id_from_gallery
                and classifier_ready.is_set()
                and frame_idx - last >= classify_every
            ):
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

        if not classifier_ready.is_set():
            cv2.putText(
                frame, "Loading model...",
                (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
            )

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
