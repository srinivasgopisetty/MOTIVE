import cv2
import os
from datetime import datetime


class EvidenceCapture:

    def __init__(self, output_dir="outputs/evidence"):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self.saved_ids = set()

    def save(self, frame, track_id, violations):

        if track_id is None:
            return None

        if not violations:
            return None

        # Save only once per motorcycle
        if track_id in self.saved_ids:
            return None

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        violation_text = "_".join(
            v.replace(" ", "_")
            for v in violations
        )

        filename = (
            f"bike_{track_id}_"
            f"{violation_text}_"
            f"{timestamp}.jpg"
        )

        path = os.path.join(
            self.output_dir,
            filename
        )

        cv2.imwrite(
            path,
            frame
        )

        self.saved_ids.add(track_id)

        print(
            f"Evidence saved: {path}"
        )

        return path