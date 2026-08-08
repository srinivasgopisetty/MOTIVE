from detector import ObjectDetector
from association import separate_objects


detector = ObjectDetector()

video_path = "data/test_videos/road.mp4"

import cv2

cap = cv2.VideoCapture(video_path)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    detections = detector.track(frame)

    motorcycles, persons = separate_objects(detections)

    for motorcycle in motorcycles:
        print(
            "Motorcycle ID:",
            motorcycle.get("track_id")
        )

    for person in persons:
        print(
            "Person ID:",
            person.get("track_id")
        )

    print("-" * 40)

cap.release()