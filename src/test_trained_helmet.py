from ultralytics import YOLO
import cv2

# Load YOUR trained model
model = YOLO("models/helmet.pt")

# Load test image
image = cv2.imread("data/test_images/helmet_test.jpg")

# Run detection
results = model(image, conf=0.25)

# Print detections
print("\nDetected Objects:\n")

for box in results[0].boxes:
    cls = int(box.cls[0])
    conf = float(box.conf[0])

    print(
        f"{model.names[cls]} "
        f"{conf:.2f}"
    )

# Show image
annotated = results[0].plot()

cv2.imshow("Helmet Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()