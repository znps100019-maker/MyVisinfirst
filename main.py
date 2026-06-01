import argparse
import json
import time

import cv2

from arm_detector import ArmDetector
from hand_detector import HandSignRecognizer


def build_args():
    # 這裡集中設定啟動參數，之後展示時可以不用改程式，只改指令。
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
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Use 0 to run until q is pressed.",
    )
    parser.add_argument(
        "--target-sign",
        default="",
        help=(
            "Target hand sign to detect, for example: Open palm, Fist, "
            "Thumbs up, OK, Number 1, Number 2, Number 3, Number 4."
        ),
    )
    parser.add_argument(
        "--target-cooldown",
        type=float,
        default=1.0,
        help="Seconds between repeated target-detected events.",
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
    # 手臂動作被計數時，可以選擇自動按空白鍵控制其他程式。
    if disable_keyboard:
        return

    try:
        import pyautogui

        pyautogui.press("space")
    except Exception as error:
        print(f"Keyboard action skipped: {error}")


def build_joint_payload(arm_joints, hand_detections, stable_status, target_sign):
    # 組成 JSON 資料，方便輸出給終端機、開發板或其他程式讀取。
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
        "stable_sign": stable_status["sign"],
        "stable": stable_status,
        "target_sign": target_sign,
        "target_detected": (
            bool(target_sign)
            and stable_status["is_stable"]
            and stable_status["sign"] == target_sign
        ),
        "arm": arm_joints,
        "hands": hands,
    }


def configure_camera(cap, args):
    # 如果指令有指定寬高，就嘗試設定攝影機解析度。
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


def main():
    args = build_args()

    # ArmDetector 負責肩膀、手肘、手腕角度；HandSignRecognizer 負責手部 21 個關節與手勢。
    arm_detector = ArmDetector()
    sign_recognizer = HandSignRecognizer()
    target_sign = sign_recognizer.normalize_sign(args.target_sign)
    counter = 0
    stage = None
    last_print_time = 0
    last_target_time = 0
    frame_count = 0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    configure_camera(cap, args)

    # 攝影機打不開時直接結束，避免後面讀取空畫面造成錯誤。
    if not cap.isOpened():
        print("Cannot open camera. Check camera permission or camera index.")
        arm_detector.close()
        sign_recognizer.close()
        return

    print("Vision system started. Press q to quit.")
    if target_sign:
        print(f"Target sign: {target_sign}")

    try:
        while True:
            # 每次迴圈讀取一張攝影機畫面，後面所有辨識都用這張影像。
            success, img = cap.read()
            if not success:
                print("Cannot read frame from camera.")
                break

            frame_count += 1

            # MediaPipe 偵測：arm_joints 是手臂角度資料，hand_detections 是手部關節與手勢資料。
            arm_joints = arm_detector.find_arm_joints(img)
            hand_detections = sign_recognizer.process(img)
            stable_status = sign_recognizer.stable_status

            # 這段用手肘角度做簡單的上下動作計數，可用來當互動控制。
            if arm_joints is not None:
                angle = arm_joints["angle"]
                if angle > 160:
                    stage = "down"
                if angle < 45 and stage == "down":
                    stage = "up"
                    counter += 1
                    maybe_press_space(args.disable_keyboard)
                    print(f"Arm counter: {counter}")

            # 如果使用者指定 --target-sign，穩定偵測到目標手勢時輸出事件。
            if target_sign and sign_recognizer.is_target_detected(target_sign):
                now = time.time()
                if now - last_target_time >= args.target_cooldown:
                    print(
                        json.dumps(
                            {
                                "event": "target_detected",
                                "timestamp": round(now, 3),
                                "target_sign": target_sign,
                                "stable": stable_status,
                            },
                            ensure_ascii=False,
                        )
                    )
                    last_target_time = now

            # print_joints 模式會把關節座標與目前手勢用 JSON 印出，適合專題展示或接其他系統。
            if args.print_joints:
                now = time.time()
                if now - last_print_time >= args.print_interval:
                    payload = build_joint_payload(
                        arm_joints,
                        hand_detections,
                        stable_status,
                        target_sign,
                    )
                    print(json.dumps(payload, ensure_ascii=False))
                    last_print_time = now

            # 非 headless 模式會開視窗，把骨架、手勢文字、目標狀態畫在畫面上。
            if not args.headless:
                draw_arm_joints(img, arm_joints)
                draw_arm_status(img, counter, stage)
                sign_recognizer.draw(img, hand_detections, target_sign)

                cv2.imshow("Graduation Project - Main Control", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # 自動測試用：跑到指定幀數就結束，避免測試時程式一直開著。
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break
    finally:
        cap.release()
        arm_detector.close()
        sign_recognizer.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
