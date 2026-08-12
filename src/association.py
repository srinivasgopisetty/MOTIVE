def separate_objects(detections):

    motorcycles = []
    persons = []

    for detection in detections:

        if detection["class"] == "motorcycle":
            motorcycles.append(detection)

        elif detection["class"] == "person":
            persons.append(detection)

    return motorcycles, persons


# ============================================================
# KEYPOINT UTILITIES
# ============================================================

def get_keypoint(point, threshold=0.30):

    if point is None:
        return None

    if len(point) < 2:
        return None

    x = float(point[0])
    y = float(point[1])

    confidence = 1.0

    if len(point) >= 3:
        confidence = float(point[2])

    if confidence < threshold:
        return None

    return (x, y, confidence)


def get_pose_features(person):

    """
    YOLO11-pose / COCO 17 keypoints.

    0  Nose
    1  Left Eye
    2  Right Eye
    3  Left Ear
    4  Right Ear
    5  Left Shoulder
    6  Right Shoulder
    7  Left Elbow
    8  Right Elbow
    9  Left Wrist
    10 Right Wrist
    11 Left Hip
    12 Right Hip
    13 Left Knee
    14 Right Knee
    15 Left Ankle
    16 Right Ankle
    """

    keypoints = person.get("keypoints")

    if not keypoints:
        return None

    if len(keypoints) < 17:
        return None

    return {
        "nose": get_keypoint(keypoints[0]),

        "left_shoulder": get_keypoint(keypoints[5]),
        "right_shoulder": get_keypoint(keypoints[6]),

        "left_elbow": get_keypoint(keypoints[7]),
        "right_elbow": get_keypoint(keypoints[8]),

        "left_wrist": get_keypoint(keypoints[9]),
        "right_wrist": get_keypoint(keypoints[10]),

        "left_hip": get_keypoint(keypoints[11]),
        "right_hip": get_keypoint(keypoints[12]),

        "left_knee": get_keypoint(keypoints[13]),
        "right_knee": get_keypoint(keypoints[14]),

        "left_ankle": get_keypoint(keypoints[15]),
        "right_ankle": get_keypoint(keypoints[16])
    }


def average_points(points):

    valid = [
        p for p in points
        if p is not None
    ]

    if not valid:
        return None

    x = sum(p[0] for p in valid) / len(valid)
    y = sum(p[1] for p in valid) / len(valid)

    return (x, y)


# ============================================================
# POSE-BASED RIDER ANALYSIS
# ============================================================

def calculate_pose_rider_score(
    motorcycle,
    person
):

    """
    Calculate how strongly the anatomical keypoints
    indicate that a person is actually riding the motorcycle.

    Lower score = better rider match.

    Returns:
        score, pose_information
    """

    features = get_pose_features(person)

    if features is None:
        return None, None

    mx1, my1, mx2, my2 = motorcycle["bbox"]

    px1, py1, px2, py2 = person["bbox"]

    mw = max(mx2 - mx1, 1)
    mh = max(my2 - my1, 1)

    # --------------------------------------------------------
    # Important body points
    # --------------------------------------------------------

    shoulders = average_points([
        features["left_shoulder"],
        features["right_shoulder"]
    ])

    hips = average_points([
        features["left_hip"],
        features["right_hip"]
    ])

    knees = average_points([
        features["left_knee"],
        features["right_knee"]
    ])

    ankles = average_points([
        features["left_ankle"],
        features["right_ankle"]
    ])

    nose = average_points([
        features["nose"]
    ])

    # --------------------------------------------------------
    # A rider should have at least a torso.
    # --------------------------------------------------------

    if shoulders is None or hips is None:

        return None, None

    # --------------------------------------------------------
    # Motorcycle center
    # --------------------------------------------------------

    motorcycle_center_x = (
        mx1 + mx2
    ) / 2

    motorcycle_center_y = (
        my1 + my2
    ) / 2

    # --------------------------------------------------------
    # 1. HIP POSITION
    #
    # A rider's hips should be close to the motorcycle
    # seating/body region.
    # --------------------------------------------------------

    hip_x, hip_y = hips

    hip_horizontal_distance = abs(
        hip_x - motorcycle_center_x
    ) / mw

    # Expanded motorcycle region.
    hip_inside = (
        mx1 - mw * 0.60
        <= hip_x
        <= mx2 + mw * 0.60
    )

    if not hip_inside:

        return None, None

    # --------------------------------------------------------
    # 2. SHOULDER POSITION
    #
    # Helps reject people completely unrelated to the bike.
    # --------------------------------------------------------

    shoulder_x, shoulder_y = shoulders

    shoulder_horizontal_distance = abs(
        shoulder_x - motorcycle_center_x
    ) / mw

    if shoulder_horizontal_distance > 1.20:

        return None, None

    # --------------------------------------------------------
    # 3. BODY HEIGHT / TORSO
    #
    # A rider should have a reasonable shoulder-to-hip
    # relationship.
    # --------------------------------------------------------

    torso_height = abs(
        hip_y - shoulder_y
    )

    if torso_height <= 0:

        return None, None

    # Reject extremely small/invalid pose geometry.
    if torso_height < ph_ratio(px1, py1, px2, py2) * 0.08:

        return None, None

    # --------------------------------------------------------
    # 4. LOWER BODY RELATIONSHIP
    #
    # Knees and ankles provide additional evidence that
    # the person is physically connected to the motorcycle.
    # --------------------------------------------------------

    lower_body_points = []

    if knees is not None:
        lower_body_points.append(knees)

    if ankles is not None:
        lower_body_points.append(ankles)

    lower_body_support = 0.0

    if lower_body_points:

        points_near_motorcycle = 0

        for x, y in lower_body_points:

            if (
                mx1 - mw * 0.75
                <= x
                <= mx2 + mw * 0.75
                and
                my1 - mh * 0.60
                <= y
                <= my2 + mh * 0.80
            ):
                points_near_motorcycle += 1

        lower_body_support = (
            points_near_motorcycle
            /
            len(lower_body_points)
        )

    # --------------------------------------------------------
    # 5. TORSO ALIGNMENT
    #
    # A rider's torso should generally remain around the
    # motorcycle centerline.
    # --------------------------------------------------------

    torso_center_x = (
        shoulder_x + hip_x
    ) / 2

    torso_horizontal_distance = abs(
        torso_center_x - motorcycle_center_x
    ) / mw

    # --------------------------------------------------------
    # 6. PERSON / MOTORCYCLE VERTICAL RELATIONSHIP
    # --------------------------------------------------------

    person_bottom = py2

    bottom_distance = abs(
        person_bottom - my2
    ) / mh

    # Keep this relatively tolerant for rear-view footage.
    if bottom_distance > 0.70:

        return None, None

    # --------------------------------------------------------
    # 7. HEAD / NOSE POSITION
    #
    # Nose is useful when visible, but it is NOT mandatory
    # because rear-view footage can hide the face.
    # --------------------------------------------------------

    head_score = 0.0

    if nose is not None:

        nose_x, nose_y = nose

        nose_horizontal_distance = abs(
            nose_x - motorcycle_center_x
        ) / mw

        head_score = min(
            nose_horizontal_distance,
            2.0
        )

    # --------------------------------------------------------
    # FINAL POSE SCORE
    #
    # Lower = stronger rider relationship.
    # --------------------------------------------------------

    score = (
        hip_horizontal_distance * 0.30
        +
        torso_horizontal_distance * 0.25
        +
        (1.0 - lower_body_support) * 0.20
        +
        min(bottom_distance, 1.5) * 0.15
        +
        head_score * 0.10
    )

    pose_information = {
        "shoulders": shoulders,
        "hips": hips,
        "knees": knees,
        "ankles": ankles,
        "nose": nose,
        "lower_body_support": lower_body_support,
        "score": score
    }

    return score, pose_information


def ph_ratio(px1, py1, px2, py2):

    """
    Person bounding-box height.
    """

    return max(
        py2 - py1,
        1
    )


# ============================================================
# RIDER ASSOCIATION
# ============================================================

def find_riders(motorcycle, persons):

    """
    Pose-assisted rider detection.

    The function:

    1. Filters people using motorcycle geometry.
    2. Requires anatomical pose information.
    3. Calculates pose-to-motorcycle relationship.
    4. Removes people that look like pedestrians/unrelated
       persons.
    5. Removes duplicate tracking IDs.
    6. Returns the final rider list.

    Therefore:

        len(riders)

    represents the estimated number of riders on the
    motorcycle.
    """

    mx1, my1, mx2, my2 = motorcycle["bbox"]

    mw = max(
        mx2 - mx1,
        1
    )

    mh = max(
        my2 - my1,
        1
    )

    motorcycle_center_x = (
        mx1 + mx2
    ) / 2

    candidates = []

    # ========================================================
    # Examine every detected person
    # ========================================================

    for person in persons:

        px1, py1, px2, py2 = person["bbox"]

        pw = px2 - px1
        ph = py2 - py1

        if pw <= 0 or ph <= 0:
            continue

        person_center_x = (
            px1 + px2
        ) / 2

        person_bottom_y = py2

        # ----------------------------------------------------
        # GEOMETRIC FILTER 1
        #
        # Person should be reasonably close to the bike's
        # horizontal region.
        #
        # This is deliberately wider than before because
        # multiple riders can occupy different horizontal
        # positions.
        # ----------------------------------------------------

        horizontal_distance = abs(
            person_center_x
            -
            motorcycle_center_x
        )

        if horizontal_distance > mw * 1.00:
            continue

        # ----------------------------------------------------
        # GEOMETRIC FILTER 2
        #
        # Person must have some vertical relationship with
        # the motorcycle.
        # ----------------------------------------------------

        vertical_overlap = (
            min(py2, my2)
            -
            max(py1, my1)
        )

        if vertical_overlap <= 0:
            continue

        overlap_ratio = (
            vertical_overlap
            /
            max(ph, 1)
        )

        if overlap_ratio < 0.15:
            continue

        # ----------------------------------------------------
        # GEOMETRIC FILTER 3
        #
        # Person should not be extremely far above/below
        # the motorcycle.
        # ----------------------------------------------------

        bottom_distance = abs(
            person_bottom_y - my2
        )

        if bottom_distance > mh * 0.70:
            continue

        # ----------------------------------------------------
        # POSE ANALYSIS
        # ----------------------------------------------------

        pose_score, pose_information = (
            calculate_pose_rider_score(
                motorcycle,
                person
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # For the new pose-assisted rider counter, a person
        # without usable anatomical keypoints is NOT counted
        # as a rider.
        #
        # This prevents ordinary nearby people from increasing
        # the rider count.
        # ----------------------------------------------------

        if pose_score is None:
            continue

        # ----------------------------------------------------
        # Pose confidence threshold
        #
        # Lower score = better pose-to-bike relationship.
        # ----------------------------------------------------

        if pose_score > 1.00:
            continue

        # ----------------------------------------------------
        # Geometric score
        # ----------------------------------------------------

        geometric_score = (
            horizontal_distance
            /
            mw
            +
            bottom_distance
            /
            mh
        )

        # ----------------------------------------------------
        # Final association score
        #
        # Pose receives the larger weight because the goal is
        # pose-assisted rider counting.
        # ----------------------------------------------------

        final_score = (
            pose_score * 0.65
            +
            geometric_score * 0.35
        )

        person["association_score"] = final_score

        person["pose_score"] = pose_score

        person["pose_information"] = (
            pose_information
        )

        candidates.append(
            (
                final_score,
                person
            )
        )

    # ========================================================
    # BEST RIDER CANDIDATES FIRST
    # ========================================================

    candidates.sort(
        key=lambda item: item[0]
    )

    riders = []

    seen_ids = set()

    # ========================================================
    # REMOVE DUPLICATE TRACK IDs
    # ========================================================

    for score, person in candidates:

        person_id = person.get(
            "track_id"
        )

        if person_id is not None:

            if person_id in seen_ids:
                continue

            seen_ids.add(
                person_id
            )

        riders.append(
            person
        )

    # ========================================================
    # RETURN FINAL POSE-BASED RIDER LIST
    # ========================================================

    return riders