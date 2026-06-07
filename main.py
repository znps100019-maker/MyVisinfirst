import os
import subprocess
import sys

def handle_non_ascii_path():
    """
    Bypass MediaPipe path encoding bug on Windows when directory contains non-ASCII characters.
    It maps the project directory to a virtual drive (e.g., Z:) and re-runs the process.
    """
    if sys.platform != "win32":
        return

    project_root = os.path.abspath(os.path.dirname(__file__))
    if any(ord(c) > 127 for c in project_root):
        # Find a free drive letter in reverse order (Z: down to D:)
        import string
        drive = None
        for letter in string.ascii_uppercase[::-1]:
            candidate = f"{letter}:"
            if not os.path.exists(candidate + "\\"):
                drive = candidate
                break

        if not drive:
            print("Error: No free drive letter found to bypass MediaPipe path bug.")
            sys.exit(1)

        # Map the drive
        subprocess.run(["subst", drive, project_root], shell=True, stdout=subprocess.DEVNULL)

        # Build paths on the virtual drive
        relative_script = os.path.relpath(os.path.abspath(__file__), project_root)
        virtual_script = os.path.join(drive, relative_script)
        virtual_python = os.path.join(drive, ".venv", "Scripts", "python.exe")

        if not os.path.exists(virtual_python):
            virtual_python = sys.executable.replace(project_root, drive)

        # Spawn the child process on the virtual drive
        args = [virtual_python, virtual_script] + sys.argv[1:]
        try:
            result = subprocess.run(args)
            returncode = result.returncode
        finally:
            # Ensure drive mapping is removed under all circumstances
            subprocess.run(["subst", drive, "/d"], shell=True, stdout=subprocess.DEVNULL)

        sys.exit(returncode)

# Run the bypass check immediately on startup before importing mediapipe/cv2
handle_non_ascii_path()

import argparse
import json
import time

import cv2

from hand_detector import HandSignRecognizer
from face_detector import FaceExpressionRecognizer


def build_args():
    # 這裡集中設定啟動參數，之後展示時可以不用改程式，只改指令。
    parser = argparse.ArgumentParser(description="Hand joint and sign detection.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    parser.add_argument("--headless", action="store_true", help="Run without a window.")
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
            "Target hand sign to detect, for example: Number 5 (Open palm), Fist, "
            "Thumbs up, OK, Number 1, Number 2, Number 3, Number 4."
        ),
    )
    parser.add_argument(
        "--target-cooldown",
        type=float,
        default=1.0,
        help="Seconds between repeated target-detected events.",
    )
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable facial expression detection.",
    )
    return parser.parse_args()


def build_joint_payload(hand_detections, stable_status, target_sign, face_expression=None):
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
        "face_expression": face_expression,
        "target_sign": target_sign,
        "target_detected": (
            bool(target_sign)
            and stable_status["is_stable"]
            and stable_status["sign"] == target_sign
        ),
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

    # HandSignRecognizer 負責手部 21 個關節與手勢。
    sign_recognizer = HandSignRecognizer()
    # FaceExpressionRecognizer 負責臉部網格與表情。
    face_recognizer = None if args.no_face else FaceExpressionRecognizer()
    
    target_sign = sign_recognizer.normalize_sign(args.target_sign)
    last_print_time = 0
    last_target_time = 0
    frame_count = 0

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    configure_camera(cap, args)

    # 攝影機打不開時直接結束，避免後面讀取空畫面造成錯誤。
    if not cap.isOpened():
        print("Cannot open camera. Check camera permission or camera index.")
        sign_recognizer.close()
        if face_recognizer:
            face_recognizer.close()
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

            # 水平翻轉影像（鏡像效果），讓操作更直覺符合鏡面反射
            img = cv2.flip(img, 1)

            frame_count += 1

            # MediaPipe 偵測：hand_detections 是手部關節與手勢資料。
            hand_detections = sign_recognizer.process(img)
            stable_status = sign_recognizer.stable_status

            # 臉部表情偵測
            face_data = None
            face_expression = None
            if face_recognizer:
                face_data = face_recognizer.process(img)
                if face_data:
                    face_expression = face_data["expression"]

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
                        hand_detections,
                        stable_status,
                        target_sign,
                        face_expression,
                    )
                    print(json.dumps(payload, ensure_ascii=False))
                    last_print_time = now

            # 非 headless 模式會開視窗，把骨架、手勢文字、目標狀態畫在畫面上。
            if not args.headless:
                if face_recognizer and face_data:
                    face_recognizer.draw(img, face_data)
                
                sign_recognizer.draw(img, hand_detections, target_sign, face_expression)

                cv2.imshow("Hand Control - Main", img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                # 判斷如果使用者點擊視窗右上角的「X」關閉按鈕，也安全退出迴圈釋放相機
                if cv2.getWindowProperty("Hand Control - Main", cv2.WND_PROP_VISIBLE) < 1:
                    break

            # 自動測試用：跑到指定幀數就結束，避免測試時程式一直開著。
            if args.max_frames > 0 and frame_count >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break
    finally:
        cap.release()
        sign_recognizer.close()
        if face_recognizer:
            face_recognizer.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
