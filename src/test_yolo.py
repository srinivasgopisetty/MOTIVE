from detector import ObjectDetector
from association import separate_objects, find_riders


def main():

    detector = ObjectDetector()

    detections = detector.detect("data/test_images/test.jpg")

    motorcycles, persons = separate_objects(detections)

    print(f"\nDetected {len(motorcycles)} motorcycles")
    print(f"Detected {len(persons)} persons\n")

    for i, motorcycle in enumerate(motorcycles):

        riders = find_riders(motorcycle, persons)

        print("=" * 40)
        print(f"Motorcycle {i + 1}")
        print(f"Bounding Box : {motorcycle['bbox']}")
        print(f"Riders Found : {len(riders)}")

        for j, rider in enumerate(riders):
            print(f"  Rider {j + 1}: {rider['bbox']}")


if __name__ == "__main__":
    main()