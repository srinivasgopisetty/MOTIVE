import cv2
from collections import defaultdict, deque
from rider_tracker import TemporalRiderTracker
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

    rider_head_center = ((rx1 + rx2) / 2, (ry1 + head_y2) / 2)

    best_index = None
    best_distance = float("inf")

    for index, helmet in enumerate(helmet_boxes):

        if index in used_helmets:
            continue

        hx1, hy1, hx2, hy2 = helmet["bbox"]

        helmet_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)

        if not (
            rx1 - rider_width * 0.20 <= helmet_center[0] <= rx2 + rider_width * 0.20
        ):
            continue

        if not (ry1 - rider_height * 0.15 <= helmet_center[1] <= head_y2):
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
        return {"label": "Unknown", "confidence": 0.0}

    strong_without = [
        p for p in history if p["label"] == "Without Helmet" and p["confidence"] >= 0.65
    ]

    strong_with = [
        p for p in history if p["label"] == "With Helmet" and p["confidence"] >= 0.50
    ]

    if len(strong_without) >= 3:

        confidence = max(p["confidence"] for p in strong_without)

        return {"label": "Without Helmet", "confidence": confidence}

    if len(strong_with) >= 2:

        confidence = max(p["confidence"] for p in strong_with)

        return {"label": "With Helmet", "confidence": confidence}

    latest = history[-1]

    return {"label": latest["label"], "confidence": latest["confidence"]}


def draw_pose_keypoints(frame, detections):

    # COCO 17-keypoint skeleton
    skeleton = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
    ]

    for detection in detections:

        if detection["class"] != "person":
            continue

        keypoints = detection.get("keypoints")

        if not keypoints:
            continue

        # ------------------------------------------------
        # Draw anatomical keypoints
        # ------------------------------------------------

        for index, point in enumerate(keypoints):

            if len(point) < 3:
                continue

            x, y, confidence = point

            if confidence < 0.20:
                continue

            x = int(x)
            y = int(y)

            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            # Keypoint index
            cv2.putText(
                frame,
                str(index),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 255, 0),
                1,
            )

        # ------------------------------------------------
        # Draw skeleton connections
        # ------------------------------------------------

        for start, end in skeleton:

            if start >= len(keypoints) or end >= len(keypoints):
                continue

            p1 = keypoints[start]
            p2 = keypoints[end]

            if len(p1) < 3 or len(p2) < 3:
                continue

            if p1[2] < 0.20 or p2[2] < 0.20:
                continue

            x1 = int(p1[0])
            y1 = int(p1[1])

            x2 = int(p2[0])
            y2 = int(p2[1])

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame


def process_video(video_path):

    detector = ObjectDetector()
    helmet_detector = HelmetDetector()
    violation_engine = ViolationEngine()
    evidence_capture = EvidenceCapture()
    rider_tracker = TemporalRiderTracker(min_hits=2, max_missing=5)

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

        # ------------------------------------------------
        # Object detection + tracking + pose keypoints
        # ------------------------------------------------

        detections = detector.track(frame)

        motorcycles, persons = separate_objects(detections)

        # ------------------------------------------------
        # Helmet detection
        # ------------------------------------------------

        helmet_result = helmet_detector.detect(original_frame)

        helmet_boxes = []

        if helmet_result is not None and len(helmet_result.boxes) > 0:

            for box in helmet_result.boxes:

                cls = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                helmet_boxes.append(
                    {
                        "class": helmet_detector.model.names[cls],
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence,
                    }
                )

        # ------------------------------------------------
        # Draw normal detections
        # ------------------------------------------------

        frame = draw_detections(frame, detections)

        # ------------------------------------------------
        # Draw anatomical keypoints
        # ------------------------------------------------

        frame = draw_pose_keypoints(frame, detections)

        # ------------------------------------------------
        # Process every motorcycle
        # ------------------------------------------------

        for motorcycle in motorcycles:

            track_id = motorcycle.get("track_id")

            if track_id is None:
                continue

            # ------------------------------------------------
            # Find riders using geometry + pose
            # ------------------------------------------------

            riders = find_riders(motorcycle, persons)
            stable_rider_count, stable_rider_ids = rider_tracker.update(
                track_id, riders
            )

            print(
                f"Bike {track_id}: "
                f"Pose Riders = {len(riders)} | "
                f"Stable Riders = {stable_rider_count} | "
                f"IDs = {sorted(stable_rider_ids)}"
            )

            for rider in riders:
                rider_id = rider.get("track_id")
                association_score = rider.get("association_score", 0.0)
                pose_score = rider.get("pose_score", 0.0)
                print(
                    f"  Rider {rider_id} | "
                    f"Association Score: {association_score:.3f} | "
                    f"Pose Score: {pose_score:.3f}"
                )

            helmet_predictions = []
            helmet_statuses = []

            used_helmets = set()

            # ------------------------------------------------
            # Process each rider
            # ------------------------------------------------

            for rider in riders:

                rider_key = rider.get("track_id")

                if rider_key is None:
                    continue

                helmet = find_helmet_for_rider(rider, helmet_boxes, used_helmets)

                if helmet is None:

                    prediction = {"label": "Unknown", "confidence": 0.0}

                else:

                    prediction = {
                        "label": helmet["class"],
                        "confidence": helmet["confidence"],
                    }

                # ------------------------------------------------
                # Temporal helmet stabilization
                # ------------------------------------------------

                stable = get_stable_status(rider_key, prediction)

                print(
                    f"Bike {track_id} "
                    f"Rider ID {rider_key}: "
                    f"{stable['label']} "
                    f"({stable['confidence']:.3f})"
                )

                helmet_predictions.append(stable)

                helmet_statuses.append(stable["label"])

            # ------------------------------------------------
            # Violation detection
            # ------------------------------------------------

            violations = violation_engine.check(
                stable_rider_count, helmet_predictions, track_id
            )

            # ------------------------------------------------
            # Evidence generation
            # ------------------------------------------------

            if violations:

                for violation in violations:

                    violation_key = (track_id, violation)

                    if violation_key not in saved_violations:

                        evidence_capture.save(frame, track_id, [violation])

                        saved_violations.add(violation_key)

            # ------------------------------------------------
            # Draw motorcycle information
            # ------------------------------------------------

            frame = draw_motorcycle_info(
                frame, motorcycle, len(riders), helmet_statuses, violations
            )

            # ------------------------------------------------
            # Draw motorcycle ID
            # ------------------------------------------------

            x1, y1, x2, y2 = motorcycle["bbox"]

            cv2.putText(
                frame,
                f"Bike ID: {track_id}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

            # ------------------------------------------------
            # Display rider association scores
            # ------------------------------------------------

            for rider in riders:

                rider_id = rider.get("track_id")

                if rider_id is None:
                    continue

                association_score = rider.get("association_score")

                if association_score is None:
                    continue

                rx1, ry1, rx2, ry2 = rider["bbox"]

                cv2.putText(
                    frame,
                    f"Rider {rider_id} " f"Assoc: {association_score:.2f}",
                    (rx1, max(20, ry1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),
                    1,
                )

        # ------------------------------------------------
        # Display
        # ------------------------------------------------

        cv2.imshow("MOTIVE - Pose Assisted Rider Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
