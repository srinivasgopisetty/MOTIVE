import cv2
from helmet_detector import HelmetDetector

detector = HelmetDetector()

image = cv2.imread("data/test_images/helmet_test.jpg")

if image is None:
    print("Could not open image.")
    exit()

result = detector.detect(image)

if result is None or len(result.boxes) == 0:
    print("No helmet detection")
else:
    for box in result.boxes:

        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = detector.model.names[cls]

        print(
            f"Class: {name} | Confidence: {conf:.3f}"
        )