import math


def get_center(bbox):
    """
    Returns the center point of a bounding box.

    bbox = (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = bbox

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return (center_x, center_y)


def euclidean_distance(point1, point2):
    """
    Returns the Euclidean distance between two points.
    """

    x1, y1 = point1
    x2, y2 = point2

    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)