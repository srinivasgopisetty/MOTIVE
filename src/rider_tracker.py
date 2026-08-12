import math


class TemporalRiderTracker:
    """
    Maintains stable physical rider identities across frames.

    Handles:
    - Temporary pose misses
    - ByteTrack ID changes
    - Rider re-identification using pose/body geometry
    """

    def __init__(
        self,
        min_hits=2,
        max_missing=8,
        reid_distance=0.35
    ):

        self.min_hits = min_hits
        self.max_missing = max_missing
        self.reid_distance = reid_distance

        self.history = {}

        self.next_physical_id = 1

        self.frame_number = 0

    # =========================================================
    # MAIN UPDATE
    # =========================================================

    def update(self, bike_id, riders):

        self.frame_number += 1

        if bike_id not in self.history:
            self.history[bike_id] = {}

        bike_history = self.history[bike_id]

        # -----------------------------------------------------
        # Track which physical riders were observed this frame
        # -----------------------------------------------------

        matched_physical_ids = set()

        # -----------------------------------------------------
        # First match using the existing ByteTrack ID
        # -----------------------------------------------------

        for rider in riders:

            track_id = rider.get("track_id")

            if track_id is None:
                continue

            matching_physical_id = None

            for physical_id, state in bike_history.items():

                if state["track_id"] == track_id:

                    matching_physical_id = physical_id
                    break

            # -------------------------------------------------
            # Existing person ID found
            # -------------------------------------------------

            if matching_physical_id is not None:

                self._update_state(
                    bike_history[
                        matching_physical_id
                    ],
                    rider
                )

                matched_physical_ids.add(
                    matching_physical_id
                )

                continue

            # -------------------------------------------------
            # New ByteTrack ID
            #
            # Attempt pose/geometry re-identification.
            # -------------------------------------------------

            reidentified_id = self._find_reidentification(
                bike_history,
                rider,
                matched_physical_ids
            )

            if reidentified_id is not None:

                self._update_state(
                    bike_history[
                        reidentified_id
                    ],
                    rider
                )

                matched_physical_ids.add(
                    reidentified_id
                )

                continue

            # -------------------------------------------------
            # Completely new physical rider
            # -------------------------------------------------

            physical_id = self.next_physical_id

            self.next_physical_id += 1

            bike_history[physical_id] = (
                self._create_state(rider)
            )

            matched_physical_ids.add(
                physical_id
            )

        # -----------------------------------------------------
        # Update missing-frame state
        # -----------------------------------------------------

        for physical_id in list(
            bike_history.keys()
        ):

            if physical_id not in matched_physical_ids:

                bike_history[
                    physical_id
                ]["missed"] += 1

        # -----------------------------------------------------
        # Remove riders that have disappeared for too long
        # -----------------------------------------------------

        for physical_id in list(
            bike_history.keys()
        ):

            state = bike_history[
                physical_id
            ]

            if state["missed"] > self.max_missing:

                del bike_history[
                    physical_id
                ]

        # -----------------------------------------------------
        # Stable physical riders
        # -----------------------------------------------------

        stable_ids = set()

        for physical_id, state in bike_history.items():

            if state["hits"] >= self.min_hits:

                stable_ids.add(
                    physical_id
                )

        return (
            len(stable_ids),
            stable_ids
        )

    # =========================================================
    # STATE CREATION
    # =========================================================

    def _create_state(self, rider):

        return {
            "track_id": rider.get("track_id"),

            "hits": 1,

            "missed": 0,

            "last_bbox": rider["bbox"],

            "last_pose": rider.get(
                "pose_information"
            ),

            "association_score": rider.get(
                "association_score",
                999.0
            )
        }

    # =========================================================
    # STATE UPDATE
    # =========================================================

    def _update_state(
        self,
        state,
        rider
    ):

        state["track_id"] = rider.get(
            "track_id"
        )

        state["hits"] += 1

        state["missed"] = 0

        state["last_bbox"] = rider[
            "bbox"
        ]

        if rider.get(
            "pose_information"
        ) is not None:

            state["last_pose"] = rider[
                "pose_information"
            ]

        state["association_score"] = rider.get(
            "association_score",
            state["association_score"]
        )

    # =========================================================
    # RE-IDENTIFICATION
    # =========================================================

    def _find_reidentification(
        self,
        bike_history,
        rider,
        matched_physical_ids
    ):

        best_id = None

        best_distance = float(
            "inf"
        )

        for physical_id, state in bike_history.items():

            # Already matched in this frame
            if physical_id in matched_physical_ids:
                continue

            # Rider must have been seen recently
            if state["missed"] > self.max_missing:
                continue

            distance = self._rider_similarity(
                state,
                rider
            )

            if distance is None:
                continue

            if distance < best_distance:

                best_distance = distance

                best_id = physical_id

        if (
            best_id is not None
            and best_distance
            <= self.reid_distance
        ):

            return best_id

        return None

    # =========================================================
    # RIDER SIMILARITY
    # =========================================================

    def _rider_similarity(
        self,
        state,
        rider
    ):

        previous_bbox = state.get(
            "last_bbox"
        )

        current_bbox = rider.get(
            "bbox"
        )

        if (
            previous_bbox is None
            or current_bbox is None
        ):
            return None

        # -----------------------------------------------------
        # Bounding-box center similarity
        # -----------------------------------------------------

        px1, py1, px2, py2 = previous_bbox

        cx1 = (px1 + px2) / 2
        cy1 = (py1 + py2) / 2

        rx1, ry1, rx2, ry2 = current_bbox

        cx2 = (rx1 + rx2) / 2
        cy2 = (ry1 + ry2) / 2

        previous_width = max(
            px2 - px1,
            1
        )

        previous_height = max(
            py2 - py1,
            1
        )

        current_width = max(
            rx2 - rx1,
            1
        )

        current_height = max(
            ry2 - ry1,
            1
        )

        scale = max(
            previous_width,
            previous_height,
            current_width,
            current_height,
            1
        )

        center_distance = math.sqrt(
            (
                (cx2 - cx1) ** 2
            )
            +
            (
                (cy2 - cy1) ** 2
            )
        ) / scale

        # -----------------------------------------------------
        # Bounding-box size similarity
        # -----------------------------------------------------

        width_ratio = abs(
            current_width
            -
            previous_width
        ) / max(
            previous_width,
            1
        )

        height_ratio = abs(
            current_height
            -
            previous_height
        ) / max(
            previous_height,
            1
        )

        size_distance = (
            width_ratio
            +
            height_ratio
        ) / 2

        # -----------------------------------------------------
        # Pose similarity
        # -----------------------------------------------------

        pose_distance = self._pose_distance(
            state.get("last_pose"),
            rider.get("pose_information")
        )

        if pose_distance is None:

            return (
                center_distance * 0.70
                +
                size_distance * 0.30
            )

        # -----------------------------------------------------
        # Combined re-identification distance
        # -----------------------------------------------------

        return (
            center_distance * 0.40
            +
            size_distance * 0.15
            +
            pose_distance * 0.45
        )

    # =========================================================
    # POSE SIMILARITY
    # =========================================================

    def _pose_distance(
        self,
        previous_pose,
        current_pose
    ):

        if (
            previous_pose is None
            or current_pose is None
        ):
            return None

        previous_points = self._extract_pose_points(
            previous_pose
        )

        current_points = self._extract_pose_points(
            current_pose
        )

        if not previous_points or not current_points:
            return None

        distances = []

        for name in previous_points:

            if name not in current_points:
                continue

            p1 = previous_points[name]
            p2 = current_points[name]

            if p1 is None or p2 is None:
                continue

            distance = math.sqrt(
                (
                    p2[0] - p1[0]
                ) ** 2
                +
                (
                    p2[1] - p1[1]
                ) ** 2
            )

            distances.append(
                distance
            )

        if not distances:
            return None

        # Normalize by torso scale when available.
        shoulder_points = []

        for pose in (
            previous_pose,
            current_pose
        ):

            shoulders = pose.get(
                "shoulders"
            )

            if shoulders is not None:

                shoulder_points.append(
                    shoulders
                )

        if len(shoulder_points) == 2:

            x1, y1 = shoulder_points[0]
            x2, y2 = shoulder_points[1]

            torso_scale = math.sqrt(
                (
                    x2 - x1
                ) ** 2
                +
                (
                    y2 - y1
                ) ** 2
            )

            torso_scale = max(
                torso_scale,
                1
            )

        else:

            torso_scale = 100.0

        return (
            sum(distances)
            /
            len(distances)
            /
            torso_scale
        )

    # =========================================================
    # POSE POINT EXTRACTION
    # =========================================================

    @staticmethod
    def _extract_pose_points(
        pose
    ):

        if pose is None:
            return {}

        points = {}

        for name in [
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle"
        ]:

            point = pose.get(
                name
            )

            if point is None:
                continue

            points[name] = (
                point[0],
                point[1]
            )

        return points

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.history.clear()

        self.next_physical_id = 1