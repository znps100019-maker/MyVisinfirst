import argparse
import json
import os
import subprocess
import sys
import time


def handle_non_ascii_path():
    """
    Bypass a MediaPipe path encoding issue on Windows by re-running from a
    temporary virtual drive when the project path contains non-ASCII characters.
    """
    if sys.platform != "win32":
        return

    project_root = os.path.abspath(os.path.dirname(__file__))
    if not any(ord(char) > 127 for char in project_root):
        return

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

    subprocess.run(["subst", drive, project_root], shell=True, stdout=subprocess.DEVNULL)

    relative_script = os.path.relpath(os.path.abspath(__file__), project_root)
    virtual_script = os.path.join(drive, relative_script)
    # Keep the interpreter that launched this process when it lives in the
    # project. A stale .venv launcher can point to a deleted base Python.
    virtual_python = sys.executable.replace(project_root, drive)
    if not os.path.exists(virtual_python):
        virtual_python = os.path.join(drive, ".venv", "Scripts", "python.exe")

    args = [virtual_python, virtual_script] + sys.argv[1:]
    try:
        result = subprocess.run(args)
        returncode = result.returncode
    finally:
        subprocess.run(["subst", drive, "/d"], shell=True, stdout=subprocess.DEVNULL)

    sys.exit(returncode)


import cv2

from core.detectors.face_detector import FaceExpressionRecognizer
from core.detectors.hand_detector import HandSignRecognizer


def build_args():
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
            "Target hand sign to detect, for example: hello, victory, salute, "
            "secret, gun, good, bad, cow, ok, i love you."
        ),
    )
    parser.add_argument(
        "--target-cooldown",
        type=float,
        default=1.0,
        help="Seconds between repeated target-detected events.",
    )
    parser.add_argument(
        "--combine-two-hands",
        action="store_true",
        help="Explicitly sum both hands into a number. Default keeps each hand separate.",
    )
    parser.add_argument(
        "--use-knn",
        action="store_true",
        help="Opt in to the existing KNN model as a global candidate.",
    )
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable facial expression detection.",
    )
    parser.add_argument(
        "--video",
        default="",
        help="Local video path or YouTube URL to process instead of camera.",
    )
    return parser.parse_args()


def build_joint_payload(hand_detections, stable_status, target_sign, face_expression=None):
    hands = [
        {
            "handedness": detection["handedness"],
            "sign": detection["sign"],
            "fingers": detection["fingers"],
            "joints": detection["joint_points"],
        }
        for detection in hand_detections
    ]

    return {
        "timestamp": round(time.time(), 3),
        "stable_sign": stable_status["sign"],
        "stable": stable_status,
        "face_expression": face_expression,
        "target_sign": target_sign,
        "target_detected": HandSignRecognizer.target_matches(
            stable_status["sign"],
            target_sign,
            stable_status["is_stable"],
        ),
        "hands": hands,
    }


def configure_camera(cap, args):
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


def open_video_source(args):
    if getattr(args, 'video', None):
        video_input = args.video
        if video_input.startswith(("http://", "https://")):
            import yt_dlp
            options = {
                "format": "bestvideo[height<=720][ext=mp4]/bestvideo[ext=mp4]/best",
                "quiet": True,
                "no_warnings": True,
            }
            
            mode = getattr(args, 'stream_mode', '2')
            if mode == '1':
                print("正在取得影片串流連結，請稍候...")
                with yt_dlp.YoutubeDL(options) as ydl:
                    try:
                        video_input = ydl.extract_info(video_input, download=False)["url"]
                    except Exception as exc:
                        print(f"Cannot open YouTube video: {exc}")
                        sys.exit(1)
            else:
                print("正在下載影片以確保穩定播放，請稍候...")
                os.makedirs("sign_language_app/temp_downloads", exist_ok=True)
                download_path = f"sign_language_app/temp_downloads/temp_video_{int(time.time())}.mp4"
                options["outtmpl"] = download_path
                with yt_dlp.YoutubeDL(options) as ydl:
                    try:
                        ydl.download([video_input])
                        video_input = download_path
                        print(f"下載完成！影片已暫存以供穩定辨識。")
                    except Exception as exc:
                        print(f"Cannot download YouTube video: {exc}")
                        sys.exit(1)
                        
        cap = cv2.VideoCapture(video_input)
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        configure_camera(cap, args)
    return cap


def print_startup(target_sign):
    print("Vision system started.")
    print("Keyboard Hotkeys:")
    print("  - Press 'q' to quit.")
    print("  - Press 'f' to toggle face mesh blue lines show/hide.")
    print("  - Press 'c' to clear current gesture history.")
    if target_sign:
        print(f"Target sign: {target_sign}")


def emit_target_event(target_sign, stable_status):
    print(
        json.dumps(
            {
                "event": "target_detected",
                "timestamp": round(time.time(), 3),
                "target_sign": target_sign,
                "stable": stable_status,
            },
            ensure_ascii=False,
        )
    )


def validate_processed_frames(frame_count, is_video):
    """Treat an opened-but-empty video as a failed run."""
    if is_video and frame_count == 0:
        raise RuntimeError("Video opened but contained zero readable frames.")


def process_video_loop(cap, sign_recognizer, face_recognizer, args, target_sign):
    last_print_time = 0
    last_target_time = 0
    frame_count = 0
    show_face_mesh = True
    target_active = False
    is_camera = not bool(args.video)

    while True:
        success, img = cap.read()
        if not success:
            print("Cannot read frame from camera or video.")
            break

        if is_camera:
            img = cv2.flip(img, 1)
        frame_count += 1

        hand_detections = sign_recognizer.process(img)
        stable_status = sign_recognizer.stable_status

        face_data = None
        face_expression = None
        if face_recognizer:
            face_data = face_recognizer.process(img)
            if face_data:
                face_expression = face_data["expression"]

        now = time.time()
        target_detected = bool(target_sign and sign_recognizer.is_target_detected(target_sign))
        if target_detected and not target_active:
            if now - last_target_time >= args.target_cooldown:
                emit_target_event(target_sign, stable_status)
                last_target_time = now
            target_active = True
        elif not target_detected:
            target_active = False

        if args.print_joints and now - last_print_time >= args.print_interval:
            payload = build_joint_payload(
                hand_detections,
                stable_status,
                target_sign,
                face_expression,
            )
            print(json.dumps(payload, ensure_ascii=False))
            last_print_time = now

        if not args.headless:
            if face_recognizer and face_data and show_face_mesh:
                face_recognizer.draw(img, face_data)

            sign_recognizer.draw(img, hand_detections, target_sign, face_expression)

            cv2.imshow("Hand Control - Main", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("f"):
                show_face_mesh = not show_face_mesh
            if key == ord("c"):
                sign_recognizer.clear_sentence()

            if cv2.getWindowProperty("Hand Control - Main", cv2.WND_PROP_VISIBLE) < 1:
                break

        if args.max_frames > 0 and frame_count >= args.max_frames:
            print(f"Reached max frames: {args.max_frames}")
            break

    validate_processed_frames(frame_count, not is_camera)
    return frame_count


def main():
    args = build_args()

    # 互動式輸入網址功能
    if not args.video:
        print("-" * 50)
        url_input = input("請貼上 YouTube 或其他網路影片連結 (或直接按 Enter 鍵開啟攝影機): ").strip()
        if url_input:
            args.video = url_input

    # 選擇串流或下載模式
    if args.video and args.video.startswith(("http://", "https://")):
        mode = input("請問要 (1) 即時串流 還是 (2) 下載後穩定播放？[預設 2]: ").strip()
        args.stream_mode = mode if mode in ["1", "2"] else "2"
    else:
        args.stream_mode = "1"
        
    print("-" * 50)

    sign_recognizer = HandSignRecognizer(
        combine_two_hands=args.combine_two_hands,
        use_knn=args.use_knn,
    )
    face_recognizer = None if args.no_face else FaceExpressionRecognizer()

    target_sign = sign_recognizer.normalize_sign(args.target_sign)

    cap = open_video_source(args)
    if not cap.isOpened():
        print("Cannot open camera or video file. Check permission or path.")
        sign_recognizer.close()
        if face_recognizer:
            face_recognizer.close()
        raise SystemExit(1)

    print_startup(target_sign)
    if not args.headless:
        cv2.namedWindow("Hand Control - Main", cv2.WINDOW_NORMAL)

    try:
        process_video_loop(cap, sign_recognizer, face_recognizer, args, target_sign)
    finally:
        cap.release()
        sign_recognizer.close()
        if face_recognizer:
            face_recognizer.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    handle_non_ascii_path()
    main()
