class ViolationEngine:

    def check(self, rider_count, helmet_statuses):

        violations = []

        # Triple riding
        if rider_count >= 3:
            violations.append("Triple Riding")

        # Helmet violation
        for status in helmet_statuses:

            if status == "Without Helmet":
                violations.append("No Helmet")
                break

        return violations