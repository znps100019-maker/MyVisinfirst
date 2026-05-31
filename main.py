import argparse
import json
import time

import cv2

from arm_detector import ArmDetector
from hand_detector import HandSignRecognizer


def build_args():
    parser = argparse.ArgumentParser(description="Hand and arm joint detection.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    parser.add_argument("--headless", action="store_true", help="Run without a window.")
    parser.add_argument(
        "--disable-keyboard",
        action="store_true",
        help="Do not press keyboard keys when arm motion is counted.",
    )
    parser.add_argument(
        "--print-joints",
        action="store_true",
        help="Print detected joints as JSON lines for board integration.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Seconds between JSON joint prints.",
    )
    return parser.parse_args()


def draw_arm_status(img, counter, stage):
    cv2.rectangle(img, (0, 0), (300, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        img,
        f"ARM COUNTER: {counter}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        img,
        f"SYSTEM STAGE: {stage}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


def draw_arm_joints(img, arm_joints):
    if arm_joints is None:
        return

    shoulder = (arm_joints["shoulder"]["x"], arm_joints["shoulder"]["y"])
    elbow = (arm_joints["elbow"]["x"], arm_joints["elbow"]["y"])
    wrist = (arm_joints["wrist"]["x"], arm_joints["wrist"]["y"])

    cv2.circle(img, shoulder, 12, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, elbow, 12, (0, 0, 255), cv2.FILLED)
    cv2.circle(img, wrist, 12, (0, 0, 255), cv2.FILLED)
    cv2.line(img, shoulder, elbow, (0, 255, 0), 3)
    cv2.line(img, elbow, wrist, (0, 255, 0), 3)
    cv2.putText(
        img,
        f'{arm_joints["angle"]} deg',
        (elbow[0] + 15, elbow[1] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )


def maybe_press_space(disable_keyboard):
    if disable_keyboard:
        return

    try:
        import pyautogui

        pyautogui.press("space")
    except Exception as error:
        print(f"Keyboard action skipped: {error}")


def build_joint_payload(arm_joints, hand_detections, stable_sign):
    hands = []
    for detection in hand_detections:
        hands.append(
            {
                "handedness": detection["handedness"],
                "sign": detection["sign"],
                "fingers": detection["fingers"],
                "joints": detection["joint_points"],
            }
        )

    return {
        "timestamp": round(time.time(), 3),
        "stable_sign": stable_sign,
        "arm": arm_joints,
        "hands": hands,
    }


def configure_camera(cap, args):
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


def main():
    args = build_args()
    arm_detector = ArmDetector()
    sign_recognizer = HandSignRecognizer()
    counter = 0
    stage = None
    last_print_time = 0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    configure_camera(cap, args)

    if not cap.isOpened():
        print("Cannot open camera. Check camera permission or camera index.")
        arm_detector.close()
        sign_recognizer.close()
        return

    print("Vision system started. Press q to quit.")

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Cannot read frame from camera.")
                break

            arm_joints = arm_detector.find_arm_joints(img)
            hand_detections = sign_recognizer.process(img)

            if arm_joints is not None:
                angle = arm_joints["angle"]
                if angle > 160:
                    stage = "down"
                if angle < 45 and stage == "down":
                    stage = "up"
                    counter += 1
                    maybe_press_space(args.disable_keyboard)
                    print(f"Arm counter: {counter}")

            if args.print_joints:
                now = time.time()
                if now - last_print_time >= args.print_interval:
                    payload = build_joint_payload(
                        arm_joints,
                        hand_detections,
                        sign_recognizer.stable_sign,
                    )
                    print(json.dumps(payload, ensure_ascii=False))
                    last_print_time = now

            if not args.headless:
                draw_arm_joints(img, arm_joints)
                draw_arm_status(img, counter, stage)
                sign_recognizer.draw(img, hand_detections)

                cv2.imshow("Graduation Project - Main Control", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        arm_detector.close()
        sign_recognizer.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
