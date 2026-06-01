import argparse

import cv2

from hand_detector import HandSignRecognizer


def build_args():
    parser = argparse.ArgumentParser(description="Preview MediaPipe hand mesh.")
    parser.add_argument("--camera", type=int, default=None, help="Camera index.")
    parser.add_argument("--scan-max", type=int, default=6, help="Indexes to auto-scan.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    return parser.parse_args()


def configure_camera(cap, args):
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None

    success, _ = cap.read()
    if not success:
        cap.release()
        return None

    return cap


def find_camera(args):
    if args.camera is not None:
        cap = open_camera(args.camera)
        return cap, args.camera

    for index in range(args.scan_max):
        cap = open_camera(index)
        if cap is not None:
            return cap, index

    return None, None


def main():
    args = build_args()
    recognizer = HandSignRecognizer()
    cap = None

    try:
        cap, camera_index = find_camera(args)
        if cap is None:
            print("No camera could be opened. Check permissions or try --camera N.")
            return

        configure_camera(cap, args)
        print(f"Camera {camera_index} opened. Press q to quit.")

        while True:
            success, frame = cap.read()
            if not success:
                print("Camera frame read failed.")
                break

            detections = recognizer.process(frame)
            recognizer.draw(frame, detections)

            cv2.imshow("Hand Mesh Test - Press q to Quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if cap is not None:
            cap.release()
        recognizer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
