import cv2

from detector import ObjectDetector
from helmet_detector import HelmetDetector
from association import separate_objects, find_riders
from violation_engine import ViolationEngine
from visualizer import draw_detections, draw_motorcycle_info


def process_video(video_path):

    detector = ObjectDetector()
    helmet_detector = HelmetDetector()
    violation_engine = ViolationEngine()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video Finished!")
            break

        detections = detector.detect(frame)

        motorcycles, persons = separate_objects(detections)

        frame = draw_detections(frame, detections)

        for motorcycle in motorcycles:

            riders = find_riders(motorcycle, persons)

            helmet_statuses = []

            for rider in riders:

                x1, y1, x2, y2 = rider["bbox"]

                rider_crop = frame[y1:y2, x1:x2]

                if rider_crop.size == 0:
                    helmet_statuses.append("Unknown")
                    continue

                helmet_result = helmet_detector.detect(rider_crop)

                helmet_label = "Unknown"

                if len(helmet_result.boxes) > 0:

                    best_box = max(
                        helmet_result.boxes,
                        key=lambda b: float(b.conf[0])
                    )

                    cls = int(best_box.cls[0])
                    helmet_label = helmet_detector.model.names[cls]

                helmet_statuses.append(helmet_label)

            violations = violation_engine.check(
                len(riders),
                helmet_statuses
            )

            frame = draw_motorcycle_info(
                frame,
                motorcycle,
                len(riders),
                helmet_statuses,
                violations
            )

        cv2.imshow("Helmet Violation Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()