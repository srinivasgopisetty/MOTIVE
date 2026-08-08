from ultralytics import YOLO


class HelmetDetector:

    def __init__(self):
        self.model = YOLO("models/helmet_200.pt")

    def detect(self, frame):

        if frame is None or frame.size == 0:
            return None

        results = self.model(
            frame,
            conf=0.20,
            imgsz=1280,
            verbose=False
        )

        return results[0]