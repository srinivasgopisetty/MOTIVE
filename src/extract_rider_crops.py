import cv2
import os
from ultralytics import YOLO


VIDEO_PATH = "data/test_videos/road.mp4"
OUTPUT_DIR = "data/rider_crops"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

saved = 0
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # Process every 5th frame
    if frame_count % 5 != 0:
        continue

    results = model(
        frame,
        classes=[0],
        conf=0.3,
        verbose=False
    )

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        width = x2 - x1
        height = y2 - y1

        # Ignore extremely small detections
        if width < 40 or height < 80:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        filename = os.path.join(
            OUTPUT_DIR,
            f"rider_{saved:03d}.jpg"
        )

        cv2.imwrite(filename, crop)

        print(
            f"Saved: {filename} "
            f"size={width}x{height}"
        )

        saved += 1

        if saved >= 20:
            break

    if saved >= 20:
        break

cap.release()

print()
print(f"Saved {saved} rider crops.")
print(f"Location: {OUTPUT_DIR}")