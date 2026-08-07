from ultralytics import YOLO


class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, image):
        """
        image can be:
        - image path (str)
        - OpenCV frame (numpy array)
        """

        results = self.model(image, conf=0.3)

        detections = []

        for box in results[0].boxes:

            cls = int(box.cls[0])
            name = self.model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": name,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0])
            })

        return detections