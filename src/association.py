def separate_objects(detections):

    motorcycles = []
    persons = []

    for detection in detections:

        if detection["class"] == "motorcycle":
            motorcycles.append(detection)

        elif detection["class"] == "person":
            persons.append(detection)

    return motorcycles, persons


def find_riders(motorcycle, persons):

    mx1, my1, mx2, my2 = motorcycle["bbox"]

    mw = mx2 - mx1
    mh = my2 - my1

    motorcycle_center_x = (mx1 + mx2) / 2

    candidates = []

    for person in persons:

        px1, py1, px2, py2 = person["bbox"]

        pw = px2 - px1
        ph = py2 - py1

        if pw <= 0 or ph <= 0:
            continue

        person_center_x = (px1 + px2) / 2
        person_bottom_y = py2

        # ------------------------------------------------
        # 1. Person must be close to the horizontal
        #    center of the motorcycle.
        # ------------------------------------------------

        horizontal_distance = abs(
            person_center_x - motorcycle_center_x
        )

        if horizontal_distance > mw * 0.65:
            continue

        # ------------------------------------------------
        # 2. Person must overlap the motorcycle vertically.
        # ------------------------------------------------

        vertical_overlap = min(py2, my2) - max(py1, my1)

        if vertical_overlap <= 0:
            continue

        overlap_ratio = vertical_overlap / ph

        if overlap_ratio < 0.20:
            continue

        # ------------------------------------------------
        # 3. Person's bottom should be close to the
        #    motorcycle's bottom.
        #
        #    This rejects pedestrians standing beside
        #    the motorcycle.
        # ------------------------------------------------

        bottom_distance = abs(
            person_bottom_y - my2
        )

        if bottom_distance > mh * 0.35:
            continue

        # ------------------------------------------------
        # 4. Person should not be completely above the bike.
        # ------------------------------------------------

        if py2 < my1 + mh * 0.25:
            continue

        # ------------------------------------------------
        # Candidate score
        # ------------------------------------------------

        score = (
            horizontal_distance / max(mw, 1)
            + bottom_distance / max(mh, 1)
        )

        candidates.append(
            (score, person)
        )

    # Sort best matching people first
    candidates.sort(
        key=lambda item: item[0]
    )

    riders = []

    seen_ids = set()

    for score, person in candidates:

        person_id = person.get("track_id")

        if person_id is not None:

            if person_id in seen_ids:
                continue

            seen_ids.add(person_id)

        riders.append(person)

    return riders