from ultralytics import YOLO


class ObjectDetector:

    def __init__(self):

        self.model = YOLO("yolo11n.pt")
        self.pose_model = YOLO("yolo11n-pose.pt")

    def detect(self, image):

        results = self.model(image, conf=0.3)

        detections = []

        for box in results[0].boxes:

            cls = int(box.cls[0])
            name = self.model.names[cls]

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            detections.append({
                "class": name,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0])
            })

        return detections

    def track(self, image):

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

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            detections.append({
                "class": name,
                "bbox": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "track_id": track_id,
                "keypoints": None
            })

        # ------------------------------------------------
        # POSE DETECTION
        # ------------------------------------------------

        pose_results = self.pose_model(
            image,
            conf=0.20,
            verbose=False
        )

        if not pose_results:
            return detections

        result = pose_results[0]

        if (
            result.boxes is None
            or result.keypoints is None
            or len(result.boxes) == 0
        ):
            return detections

        pose_boxes = result.boxes
        pose_keypoints = result.keypoints.data

        for pose_index, pose_box in enumerate(pose_boxes):

            # Pose model is COCO person detection.
            pose_cls = int(pose_box.cls[0])

            if pose_cls != 0:
                continue

            px1, py1, px2, py2 = map(
                int,
                pose_box.xyxy[0]
            )

            keypoints = (
                pose_keypoints[pose_index]
                .cpu()
                .numpy()
                .tolist()
            )

            best_detection = None
            best_iou = 0.0

            # ------------------------------------------------
            # Match pose person to existing tracked person
            # ------------------------------------------------

            for detection in detections:

                if detection["class"] != "person":
                    continue

                dx1, dy1, dx2, dy2 = detection["bbox"]

                iou = self._calculate_iou(
                    (px1, py1, px2, py2),
                    (dx1, dy1, dx2, dy2)
                )

                if iou > best_iou:

                    best_iou = iou
                    best_detection = detection

            if (
                best_detection is not None
                and best_iou >= 0.20
            ):

                best_detection["keypoints"] = keypoints

        return detections

    @staticmethod
    def _calculate_iou(box_a, box_b):

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_width = max(
            0,
            ix2 - ix1
        )

        intersection_height = max(
            0,
            iy2 - iy1
        )

        intersection = (
            intersection_width
            * intersection_height
        )

        area_a = max(
            0,
            ax2 - ax1
        ) * max(
            0,
            ay2 - ay1
        )

        area_b = max(
            0,
            bx2 - bx1
        ) * max(
            0,
            by2 - by1
        )

        union = (
            area_a
            + area_b
            - intersection
        )

        if union <= 0:
            return 0.0

        return intersection / union