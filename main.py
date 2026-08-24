import argparse
import os
import threading
import time
import cv2
import numpy as np

from detector import FaceDetector
from classifier import GenderClassifier
from tracker import CentroidTracker
from reid import PersonDatabase

CAPTURES_DIR = "captures"


def save_face(frame: np.ndarray, box: tuple[int, int, int, int], person_id: int, gender: str) -> None:
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    x, y, w, h = box
    # add 20% padding so the crop includes forehead and chin
    pad_x, pad_y = int(w * 0.2), int(h * 0.2)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CAPTURES_DIR, f"person_{person_id:03d}_{gender}_{ts}.jpg")
    cv2.imwrite(filename, crop)
    print(f"[capture] saved {filename}")


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
    # position_radius: how far (px) a returning face can be from its last known spot
    # memory_frames: how many frames (at ~30fps: 300 ≈ 10 seconds) a person is remembered
    person_db = PersonDatabase(position_radius=200, memory_frames=300)

    classifier: GenderClassifier | None = None
    classifier_ready = threading.Event()

    def _load_models() -> None:
        nonlocal classifier
        from deepface import DeepFace
        try:
            DeepFace.build_model("Gender")  # force weights into memory before first classify
        except Exception:
            pass
        classifier = GenderClassifier()
        classifier_ready.set()

    threading.Thread(target=_load_models, daemon=True).start()

    id_last_frame: dict[int, int] = {}      # centroid ID → last classified frame
    centroid_to_person: dict[int, int] = {} # centroid ID → person DB ID
    counted_persons: set[int] = set()       # person DB IDs already in totals
    totals: dict[str, int] = {"Man": 0, "Woman": 0}
    frame_idx = 0

    print("Press Q to quit.  (models loading in background...)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        faces = detector.detect(frame)
        tracked = tracker.update(faces)

        # Clean up deregistered centroid IDs (PersonDatabase entries survive forever)
        for old_id in tracker.just_deregistered:
            centroid_to_person.pop(old_id, None)
            id_last_frame.pop(old_id, None)

        for obj_id, centroid in tracked.items():
            box = nearest_box(centroid, faces)
            if box is None:
                continue

            # Step 1 — resolve person identity via position (immediate, no model needed)
            if obj_id not in centroid_to_person:
                person_id, known_gender, _ = person_db.identify(centroid, frame_idx)
                centroid_to_person[obj_id] = person_id
            else:
                person_id = centroid_to_person[obj_id]
                person_db.update_position(person_id, centroid, frame_idx)

            # Step 2 — classify gender (gated until models finish loading)
            # Classify immediately if no gender yet; otherwise throttle to every N frames
            no_gender_yet = person_db.get_gender(person_id) is None
            last = id_last_frame.get(obj_id, -classify_every)
            due = no_gender_yet or (frame_idx - last >= classify_every)
            if classifier_ready.is_set() and due:
                gender = classifier.classify(frame, box)
                if gender:
                    id_last_frame[obj_id] = frame_idx
                    person_db.set_gender(person_id, gender)
                    if person_id not in counted_persons:
                        counted_persons.add(person_id)
                        totals[gender] = totals.get(gender, 0) + 1
                        save_face(frame, box, person_id, gender)

        # Draw bounding boxes and labels
        for obj_id, centroid in tracked.items():
            person_id = centroid_to_person.get(obj_id, obj_id)
            gender = person_db.get_gender(person_id) or "?"
            color = (255, 120, 50) if gender == "Man" else (50, 120, 255)
            box = nearest_box(centroid, faces)
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, f"ID:{person_id}  {gender}",
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
                frame, "Loading models...",
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
