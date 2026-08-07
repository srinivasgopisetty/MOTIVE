import cv2


def draw_detections(frame, detections):

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]
        label = detection["class"]

        color = (0, 255, 0)

        if label == "motorcycle":
            color = (255, 0, 0)

        elif label == "person":
            color = (0, 255, 255)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return frame


def draw_motorcycle_info(
    frame,
    motorcycle,
    rider_count,
    helmet_statuses,
    violations
):

    x1, y1, x2, y2 = motorcycle["bbox"]

    line = y1 - 10

    cv2.putText(
        frame,
        f"Riders : {rider_count}",
        (x1, line),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    line += 22

    if len(helmet_statuses):

        text = ", ".join(helmet_statuses)

        cv2.putText(
            frame,
            text,
            (x1, line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    line += 22

    for violation in violations:

        cv2.putText(
            frame,
            "⚠ " + violation,
            (x1, line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        line += 22

    return frame