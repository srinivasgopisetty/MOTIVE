from utils.geometry import get_center, euclidean_distance


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

    riders = []

    motorcycle_center = get_center(motorcycle["bbox"])

    mx1, my1, mx2, my2 = motorcycle["bbox"]

    for person in persons:

        person_center = get_center(person["bbox"])

        px, py = person_center

        horizontal_overlap = mx1 <= px <= mx2

        above_motorcycle = py < my2

        distance = euclidean_distance(
            motorcycle_center,
            person_center
        )

        if horizontal_overlap and above_motorcycle and distance < 250:
            riders.append(person)

    return riders