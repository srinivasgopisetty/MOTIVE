import cv2
from collections import defaultdict, deque

from detector import ObjectDetector
from helmet_detector import HelmetDetector
from association import separate_objects, find_riders
from violation_engine import ViolationEngine
from evidence import EvidenceCapture
from visualizer import draw_detections, draw_motorcycle_info


rider_history = defaultdict(lambda: deque(maxlen=10))
saved_violations = set()


def find_helmet_for_rider(rider, helmet_boxes, used_helmets):

    rx1, ry1, rx2, ry2 = rider["bbox"]

    rider_height = ry2 - ry1
    rider_width = rx2 - rx1

    head_y2 = ry1 + int(rider_height * 0.60)

    rider_head_center = (
        (rx1 + rx2) / 2,
        (ry1 + head_y2) / 2
    )

    best_index = None
    best_distance = float("inf")

    for index, helmet in enumerate(helmet_boxes):

        if index in used_helmets:
            continue

        hx1, hy1, hx2, hy2 = helmet["bbox"]

        helmet_center = (
            (hx1 + hx2) / 2,
            (hy1 + hy2) / 2
        )

        if not (
            rx1 - rider_width * 0.20
            <= helmet_center[0]
            <= rx2 + rider_width * 0.20
        ):
            continue

        if not (
            ry1 - rider_height * 0.15
            <= helmet_center[1]
            <= head_y2
        ):
            continue

        distance = (
            (helmet_center[0] - rider_head_center[0]) ** 2
            + (helmet_center[1] - rider_head_center[1]) ** 2
        ) ** 0.5

        if distance < best_distance:
            best_distance = distance
            best_index = index

    if best_index is None:
        return None

    used_helmets.add(best_index)

    return helmet_boxes[best_index]


def get_stable_status(rider_key, prediction):

    history = rider_history[rider_key]

    if prediction["label"] != "Unknown":
        history.append(prediction)

    if not history:
        return {
            "label": "Unknown",
            "confidence": 0.0
        }

    strong_without = [
        p for p in history
        if p["label"] == "Without Helmet"
        and p["confidence"] >= 0.65
    ]

    strong_with = [
        p for p in history
        if p["label"] == "With Helmet"
        and p["confidence"] >= 0.50
    ]

    if len(strong_without) >= 3:

        confidence = max(
            p["confidence"]
            for p in strong_without
        )

        return {
            "label": "Without Helmet",
            "confidence": confidence
        }

    if len(strong_with) >= 2:

        confidence = max(
            p["confidence"]
            for p in strong_with
        )

        return {
            "label": "With Helmet",
            "confidence": confidence
        }

    latest = history[-1]

    return {
        "label": latest["label"],
        "confidence": latest["confidence"]
    }


def process_video(video_path):

    detector = ObjectDetector()
    helmet_detector = HelmetDetector()
    violation_engine = ViolationEngine()
    evidence_capture = EvidenceCapture()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video Finished!")
            break

        original_frame = frame.copy()

        detections = detector.track(frame)

        motorcycles, persons = separate_objects(detections)

        helmet_result = helmet_detector.detect(
            original_frame
        )

        helmet_boxes = []

        if (
            helmet_result is not None
            and len(helmet_result.boxes) > 0
        ):

            for box in helmet_result.boxes:

                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                helmet_boxes.append({
                    "class": helmet_detector.model.names[cls],
                    "bbox": (x1, y1, x2, y2),
                    "confidence": confidence
                })

        frame = draw_detections(
            frame,
            detections
        )

        for motorcycle in motorcycles:

            track_id = motorcycle.get("track_id")

            if track_id is None:
                continue

            riders = find_riders(
                motorcycle,
                persons
            )

            helmet_predictions = []
            helmet_statuses = []

            used_helmets = set()

            for rider in riders:

                rider_key = rider.get("track_id")

                if rider_key is None:
                    continue

                helmet = find_helmet_for_rider(
                    rider,
                    helmet_boxes,
                    used_helmets
                )

                if helmet is None:

                    prediction = {
                        "label": "Unknown",
                        "confidence": 0.0
                    }

                else:

                    prediction = {
                        "label": helmet["class"],
                        "confidence": helmet["confidence"]
                    }

                stable = get_stable_status(
                    rider_key,
                    prediction
                )

                print(
                    f"Bike {track_id} "
                    f"Rider ID {rider_key}: "
                    f"{stable['label']} "
                    f"({stable['confidence']:.3f})"
                )

                helmet_predictions.append(stable)
                helmet_statuses.append(stable["label"])

            violations = violation_engine.check(
                len(riders),
                helmet_predictions,
                track_id
            )

            if violations:

                for violation in violations:

                    violation_key = (
                        track_id,
                        violation
                    )

                    if violation_key not in saved_violations:

                        evidence_capture.save(
                            frame,
                            track_id,
                            [violation]
                        )

                        saved_violations.add(
                            violation_key
                        )

            frame = draw_motorcycle_info(
                frame,
                motorcycle,
                len(riders),
                helmet_statuses,
                violations
            )

            x1, y1, x2, y2 = motorcycle["bbox"]

            cv2.putText(
                frame,
                f"Bike ID: {track_id}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        cv2.imshow(
            "Helmet Violation Detection",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()