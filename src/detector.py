from ultralytics import YOLO


class ObjectDetector:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, image):
        """
        Normal object detection.
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

    def track(self, image):
        """
        Object detection + ByteTrack tracking.
        """

        results = self.model.track(
            image,
            conf=0.3,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []

        for box in results[0].boxes:

            cls = int(box.cls[0])
            name = self.model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            detections.append({
                "class": name,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "track_id": track_id
            })

        return detections