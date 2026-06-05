import argparse

import cv2


def build_args():
    parser = argparse.ArgumentParser(description="Scan available camera indexes.")
    parser.add_argument(
        "--max",
        type=int,
        default=10,
        help="Number of camera indexes to test, starting from 0.",
    )
    return parser.parse_args()


def check_available_cameras(max_to_test=10):
    available_cameras = []

    print("Scanning camera indexes...")
    print("-" * 40)

    for index in range(max_to_test):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        try:
            if not cap.isOpened():
                continue

            success, _ = cap.read()
            if not success:
                print(f"[WARN] Camera index {index} opened but no frame was read.")
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[OK] Camera index {index}: {width} x {height}")
            available_cameras.append(index)
        finally:
            cap.release()

    print("-" * 40)
    if available_cameras:
        print(f"Available camera indexes: {available_cameras}")
    else:
        print("No available camera was found.")

    return available_cameras


if __name__ == "__main__":
    args = build_args()
    check_available_cameras(args.max)
