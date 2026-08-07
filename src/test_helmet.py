from ultralytics import YOLO
import cv2

# Load helmet model
model = YOLO("models/helmet.pt")

# Read test image
image = cv2.imread("data/test_images/test.jpg")

# Run detection
results = model(image, conf=0.3)

# Draw detections
annotated = results[0].plot()

# Show image
cv2.imshow("Helmet Detection", annotated)

cv2.waitKey(0)
cv2.destroyAllWindows()