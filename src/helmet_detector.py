from ultralytics import YOLO


class HelmetDetector:
    def __init__(self):
        self.model = YOLO("models/helmet.pt")

    def detect(self, image):
        """
        Detect helmets in an image.

        Returns:
            results[0]
        """
        results = self.model(image, conf=0.25, verbose=False)
        return results[0]