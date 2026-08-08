class ViolationEngine:

    def __init__(self):
        self.triple_count = {}

    def check(self, rider_count, helmet_predictions, track_id=None):

        violations = []

        # Triple riding
        if track_id is not None:

            if track_id not in self.triple_count:
                self.triple_count[track_id] = 0

            if rider_count >= 3:
                self.triple_count[track_id] += 1
            else:
                self.triple_count[track_id] = 0

            # Require 5 consecutive frames
            if self.triple_count[track_id] >= 5:
                violations.append("Triple Riding")

        elif rider_count >= 3:
            violations.append("Triple Riding")

        # No Helmet
        for prediction in helmet_predictions:

            label = prediction.get("label")
            confidence = prediction.get("confidence", 0.0)

            if (
                label == "Without Helmet"
                and confidence >= 0.65
            ):
                violations.append("No Helmet")
                break

        return violations