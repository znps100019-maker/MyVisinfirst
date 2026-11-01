import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def configure_console_encoding():
    """Keep Traditional Chinese diagnostics readable on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError):
                pass


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
    virtual_python = sys.executable.replace(project_root, drive)
    if not os.path.exists(virtual_python):
        virtual_python = os.path.join(drive, ".venv", "Scripts", "python.exe")

    args = [virtual_python, virtual_script] + sys.argv[1:]
    child_env = os.environ.copy()
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


if __name__ == "__main__":
    handle_non_ascii_path()


import cv2

from core.detectors.face_detector import FaceExpressionRecognizer
from core.detectors.hand_detector import HandSignRecognizer


def _ascii_path(path):
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
    import mediapipe.python.solution_base as solution_base
    source_file = os.path.abspath(solution_base.__file__)
    source_site_packages = os.path.dirname(os.path.dirname(os.path.dirname(source_file)))
    if all(ord(char) < 128 for char in source_file):
        return source_site_packages

    target_site_packages = _ascii_path(os.path.join(tempfile.gettempdir(), "myvisinfirst_mediapipe"))
    source_modules = os.path.join(source_site_packages, "mediapipe", "modules")
    target_modules = os.path.join(target_site_packages, "mediapipe", "modules")
    required_hand_graph = os.path.join(target_modules, "hand_landmark", "hand_landmark_tracking_cpu.binarypb")
    if not os.path.exists(required_hand_graph):
        if not os.path.isdir(source_modules):
            raise FileNotFoundError(f"找不到 MediaPipe 模型資源：{source_modules}")
        os.makedirs(os.path.dirname(target_modules), exist_ok=True)
        shutil.copytree(source_modules, target_modules, dirs_exist_ok=True)

    fake_solution_file = os.path.join(target_site_packages, "mediapipe", "python", "solution_base.py")
    os.makedirs(os.path.dirname(fake_solution_file), exist_ok=True)
    solution_base.__file__ = fake_solution_file
    return target_site_packages


def build_args():
    parser = argparse.ArgumentParser(description="Unified camera and hand-sign detection (Pure KNN).")
    parser.add_argument("source", nargs="?", default="", help="Optional video path. Omit to use camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--width", type=int, default=0, help="Optional camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional camera height.")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--print-joints", action="store_true")
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--target-sign", default="")
    parser.add_argument("--target-cooldown", type=float, default=1.0)
    
    hand_mode = parser.add_mutually_exclusive_group()
    hand_mode.add_argument("--combine-two-hands", dest="combine_two_hands", action="store_true", default=True)
    hand_mode.add_argument("--separate-hands", dest="combine_two_hands", action="store_false")
    
    parser.add_argument("--no-face", action="store_true")
    parser.add_argument("--video", "--input", dest="video", default="")
    parser.add_argument("--stream-mode", choices=("download", "stream"), default="download")
    
    args = parser.parse_args()
    if args.source and args.video:
        parser.error("只能使用位置參數或 --video/--input 其中一種影片來源")
    if args.source:
        args.video = args.source
    return args


def resolve_interactive_source(args, input_func=input):
    if args.video or not sys.stdin.isatty():
        return args
    print("-" * 50)
    video_input = input_func("請貼上影片連結；直接按 Enter 開啟攝像頭做手語偵測：").strip()
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
        "target_detected": HandSignRecognizer.target_matches(stable_status["sign"], target_sign, stable_status["is_stable"]),
        "hands": hands,
    }


def configure_camera(cap, args):
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)


def open_camera_source(camera_index):
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

    print(f"攝像頭 {camera_index} 已被找到但無法讀取影像。請確認 Windows 相機權限。", flush=True)
    return None


def open_video_source(args):
    if getattr(args, 'video', None):
        video_input = args.video.strip()
        if not video_input:
            return cv2.VideoCapture()
        
        # 影片處理邏輯保持不變...
        # (略，與你原本的 main.py 相同)
        if not video_input.startswith(("http://", "https://")) and not os.path.isfile(video_input):
            print(f"找不到影片檔案：{video_input}", flush=True)
            return cv2.VideoCapture()
        return cv2.VideoCapture(video_input)
    else:
        camera_indices = [args.camera]
        if args.camera == 0:
            camera_indices.extend([1, 2, 3])
        for camera_index in camera_indices:
            cap = open_camera_source(camera_index)
            if cap is not None:
                configure_camera(cap, args)
                args.camera = camera_index
                return cap
        return cv2.VideoCapture()


def print_startup(target_sign):
    print("Vision system started.")
    print("Keyboard Hotkeys:")
    print("  - Press 'q' to quit.")
    print("  - Press 'f' to toggle face mesh lines.")
    print("  - Press 'c' to clear gesture history.")
    if target_sign:
        print(f"Target sign: {target_sign}")


def emit_target_event(target_sign, stable_status):
    print(json.dumps({"event": "target_detected", "timestamp": round(time.time(), 3), "target_sign": target_sign, "stable": stable_status}, ensure_ascii=False))


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
            break

        if is_camera:
            img = cv2.flip(img, 1)
        frame_count += 1

        face_data = None
        face_expression = None
        face_landmarks = None
        if face_recognizer:
            face_data = face_recognizer.process(img)
            if face_data:
                face_expression = face_data["expression"]
                face_landmarks = face_data.get("landmarks")

        hand_detections = sign_recognizer.process(img, face_landmarks=face_landmarks)
        stable_status = sign_recognizer.stable_status

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
            payload = build_joint_payload(hand_detections, stable_status, target_sign, face_expression)
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
            break


def main():
    configure_console_encoding()
    args = resolve_interactive_source(build_args())

    print("-" * 50)
    print(f"輸入來源：{'影片' if args.video else f'攝像頭 {args.camera}'}", flush=True)
    print("正在初始化 MediaPipe 手部辨識...", flush=True)
    print("辨識模式：純 KNN 動態手語模型 (已停用規則式數字)", flush=True)

    try:
        resource_root = prepare_mediapipe_resources()
        sign_recognizer = HandSignRecognizer(
            combine_two_hands=args.combine_two_hands,
            use_knn=True,
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
        print("Cannot open video/camera source.")
        sign_recognizer.close()
        if face_recognizer:
            face_recognizer.close()
        raise SystemExit(1)

    print_startup(target_sign)
    if not args.headless:
        try:
            cv2.namedWindow("Hand Control - Main", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Hand Control - Main", max(640, args.window_width), max(480, args.window_height))
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