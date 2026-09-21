import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
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
    child_env = os.environ.copy()
    # MediaPipe resolves graph files from sys.path. Remap environment paths as
    # well, otherwise PYTHONPATH can keep pointing to the non-ASCII C: path.
    for variable in ("PYTHONPATH", "PATH", "VIRTUAL_ENV", "PYTHONHOME"):
        value = child_env.get(variable)
        if value:
            child_env[variable] = value.replace(project_root, drive)
    try:
        result = subprocess.run(args, env=child_env)
        returncode = result.returncode
    finally:
        subprocess.run(["subst", drive, "/d"], shell=True, stdout=subprocess.DEVNULL)

    sys.exit(returncode)


# This must run before importing MediaPipe. MediaPipe builds its resource root
# from the imported module path, so remapping afterwards is too late.
if __name__ == "__main__":
    handle_non_ascii_path()


import cv2

from core.detectors.face_detector import FaceExpressionRecognizer
from core.detectors.hand_detector import HandSignRecognizer


def _ascii_path(path):
    """Return an ASCII Windows path when the filesystem supports short names."""
    if all(ord(char) < 128 for char in path):
        return path
    if sys.platform == "win32":
        try:
            get_short_path = ctypes.windll.kernel32.GetShortPathNameW
            get_short_path.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
            get_short_path.restype = ctypes.c_uint32
            buffer = ctypes.create_unicode_buffer(32768)
            length = get_short_path(path, buffer, len(buffer))
            if length and all(ord(char) < 128 for char in buffer.value):
                return buffer.value
        except (AttributeError, OSError):
            pass
    return os.path.join(os.environ.get("SystemDrive", "C:"), "mv_mediapipe_resources")


def prepare_mediapipe_resources():
    """Make MediaPipe graph/model resources readable from an ASCII path.

    MediaPipe derives its resource directory from solution_base.__file__. On
    Windows, some installations fail to open that path when the project lives
    under a non-ASCII OneDrive directory. Copying only the package resources
    keeps the installed Python code unchanged while fixing graph loading.
    """
    import mediapipe.python.solution_base as solution_base

    source_file = os.path.abspath(solution_base.__file__)
    source_site_packages = os.path.dirname(
        os.path.dirname(os.path.dirname(source_file))
    )
    if all(ord(char) < 128 for char in source_file):
        return source_site_packages

    target_site_packages = _ascii_path(
        os.path.join(tempfile.gettempdir(), "myvisinfirst_mediapipe")
    )
    source_modules = os.path.join(source_site_packages, "mediapipe", "modules")
    target_modules = os.path.join(target_site_packages, "mediapipe", "modules")
    required_hand_graph = os.path.join(
        target_modules, "hand_landmark", "hand_landmark_tracking_cpu.binarypb"
    )
    if not os.path.exists(required_hand_graph):
        if not os.path.isdir(source_modules):
            raise FileNotFoundError(f"找不到 MediaPipe 模型資源：{source_modules}")
        os.makedirs(os.path.dirname(target_modules), exist_ok=True)
        shutil.copytree(source_modules, target_modules, dirs_exist_ok=True)

    fake_solution_file = os.path.join(
        target_site_packages, "mediapipe", "python", "solution_base.py"
    )
    os.makedirs(os.path.dirname(fake_solution_file), exist_ok=True)
    solution_base.__file__ = fake_solution_file
    return target_site_packages


def build_args():
    parser = argparse.ArgumentParser(
        description="Unified camera, local-video, and YouTube hand-sign detection."
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="",
        help="Optional local video path or YouTube URL. Omit it to use the camera.",
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    parser.add_argument(
        "--window-width", type=int, default=1280,
        help="Display window width in pixels.",
    )
    parser.add_argument(
        "--window-height", type=int, default=720,
        help="Display window height in pixels.",
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="Show the camera detection window in fullscreen mode.",
    )
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
    hand_mode = parser.add_mutually_exclusive_group()
    hand_mode.add_argument(
        "--combine-two-hands",
        dest="combine_two_hands",
        action="store_true",
        default=True,
        help="Combine both hands into one number (default).",
    )
    hand_mode.add_argument(
        "--separate-hands",
        dest="combine_two_hands",
        action="store_false",
        help="Keep two hands as separate hand classifications.",
    )
    knn_mode = parser.add_mutually_exclusive_group()
    knn_mode.add_argument(
        "--use-knn",
        dest="use_knn",
        action="store_true",
        default=True,
        help="Use KNN sign language recognition model (default).",
    )
    knn_mode.add_argument(
        "--no-knn",
        dest="use_knn",
        action="store_false",
        help="Disable KNN model and use rule-based hand shapes only.",
    )
    parser.add_argument(
        "--no-face",
        action="store_true",
        help="Disable facial expression detection.",
    )
    parser.add_argument(
        "--video", "--input",
        dest="video",
        default="",
        help="Local video path or YouTube URL to process instead of camera.",
    )
    parser.add_argument(
        "--stream-mode",
        choices=("download", "stream"),
        default="download",
        help="YouTube mode: download locally for stability or use a live stream URL.",
    )
    args = parser.parse_args()
    if args.source and args.video:
        parser.error("只能使用位置參數或 --video/--input 其中一種影片來源")
    if args.source:
        args.video = args.source
    return args


def resolve_interactive_source(args, input_func=input):
    """Ask for an optional video only in an interactive terminal.

    An empty answer intentionally keeps ``args.video`` empty, which means the
    main loop opens the camera and detects the person in front of it.
    """
    if args.video or not sys.stdin.isatty():
        return args
    print("-" * 50)
    video_input = input_func(
        "請貼上影片/YouTube 連結；直接按 Enter 開啟攝像頭做人員偵測："
    ).strip()
    if video_input:
        args.video = video_input
    return args


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


def open_camera_source(camera_index):
    """Open a camera and verify that it can provide at least one frame."""
    backends = [("DirectShow", cv2.CAP_DSHOW)]
    media_foundation = getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)
    if media_foundation not in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        backends.append(("Media Foundation", media_foundation))
    if cv2.CAP_ANY not in (cv2.CAP_DSHOW, media_foundation):
        backends.append(("default", cv2.CAP_ANY))

    for backend_name, backend in backends:
        print(f"正在開啟攝像頭 {camera_index} ({backend_name})...", flush=True)
        cap = cv2.VideoCapture(camera_index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        success, _ = cap.read()
        if success:
            print(f"攝像頭 {camera_index} 已開啟。", flush=True)
            return cap
        cap.release()

    print(
        f"攝像頭 {camera_index} 已被找到但無法讀取影像。"
        "請確認 Windows 相機權限，並關閉其他使用攝像頭的程式。",
        flush=True,
    )
    return None


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
            
            if args.stream_mode == "stream":
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
        camera_indices = [args.camera]
        # A USB camera or virtual camera is often exposed as index 1+.
        # Keep an explicit non-zero --camera request limited to that index.
        if args.camera == 0:
            camera_indices.extend([1, 2, 3])
        for camera_index in camera_indices:
            cap = open_camera_source(camera_index)
            if cap is not None:
                configure_camera(cap, args)
                args.camera = camera_index
                return cap
        return cv2.VideoCapture()
    return cap


def print_startup(target_sign):
    print("Vision system started.")
    print("Keyboard Hotkeys:")
    print("  - Press 'q' to quit.")
    print("  - Press 'f' to toggle face mesh blue lines show/hide.")
    print("  - Press 'c' to clear current gesture history.")
    print("  - Two-hand combine mode: enabled.")
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
    args = resolve_interactive_source(build_args())

    print("-" * 50)
    if args.video:
        print(f"輸入來源：{args.video}", flush=True)
    else:
        print(f"輸入來源：攝像頭 {args.camera}", flush=True)
    print("正在初始化 MediaPipe 手部辨識...", flush=True)
    print(
        "雙手合併模式：啟用" if args.combine_two_hands else "雙手合併模式：停用",
        flush=True,
    )
    print(
        "KNN 手語模型：啟用 (優先進行手語詞彙辨識)"
        if args.use_knn
        else "KNN 手語模型：停用 (僅使用靜態手形規則)",
        flush=True,
    )

    try:
        resource_root = prepare_mediapipe_resources()
        if resource_root:
            print(f"MediaPipe 資源路徑：{resource_root}", flush=True)
        sign_recognizer = HandSignRecognizer(
            combine_two_hands=args.combine_two_hands,
            use_knn=args.use_knn,
        )
        face_recognizer = None if args.no_face else FaceExpressionRecognizer()
    except Exception as exc:
        print(f"辨識器初始化失敗：{type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(1)
    if sign_recognizer.knn_samples:
        vocab_count = len(set(s["label"] for s in sign_recognizer.knn_samples))
        print(f"已載入 KNN 模型：{len(sign_recognizer.knn_samples)} 筆樣本 ({vocab_count} 種手語詞彙)", flush=True)
    print("MediaPipe 初始化完成。", flush=True)

    target_sign = sign_recognizer.normalize_sign(args.target_sign)

    cap = open_video_source(args)
    if not cap.isOpened():
        if args.video:
            print(f"Cannot open video source: {args.video}")
        else:
            print(
                f"Cannot open camera index {args.camera}. "
                "Check Windows camera permission, that another app is not using it, "
                "or try --camera 1."
            )
        sign_recognizer.close()
        if face_recognizer:
            face_recognizer.close()
        raise SystemExit(1)

    print_startup(target_sign)
    if not args.headless:
        try:
            cv2.namedWindow("Hand Control - Main", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(
                "Hand Control - Main",
                max(640, args.window_width),
                max(480, args.window_height),
            )
            if args.fullscreen:
                cv2.setWindowProperty(
                    "Hand Control - Main",
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
        except cv2.error as exc:
            print(f"無法建立即時影像視窗：{exc}", flush=True)
            cap.release()
            sign_recognizer.close()
            if face_recognizer:
                face_recognizer.close()
            raise SystemExit(1)

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
    main()
