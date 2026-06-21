import argparse
import json
import os
import subprocess
import sys
from collections import Counter, deque


def handle_non_ascii_path():
    if sys.platform != "win32":
        return

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
    virtual_python = os.path.join(drive, ".venv", "Scripts", "python.exe")
    if not os.path.exists(virtual_python):
        virtual_python = sys.executable.replace(project_root, drive)

    try:
        result = subprocess.run([virtual_python, virtual_script] + sys.argv[1:])
        returncode = result.returncode
    finally:
        subprocess.run(["subst", drive, "/d"], shell=True, stdout=subprocess.DEVNULL)

    sys.exit(returncode)


handle_non_ascii_path()

import cv2
import mediapipe as mp
import yt_dlp

try:
    from knn import classify_knn
    from labels import display_label
    from landmarks import mediapipe_landmarks_to_list, normalize_landmarks
    from text_overlay import draw_panel, draw_text
except ImportError:
    from sign_language_app.knn import classify_knn
    from sign_language_app.labels import display_label
    from sign_language_app.landmarks import mediapipe_landmarks_to_list, normalize_landmarks
    from sign_language_app.text_overlay import draw_panel, draw_text


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def build_args():
    parser = argparse.ArgumentParser(description="Recognize trained sign-language labels from a video.")
    parser.add_argument("--input", required=True, help="Local video path or YouTube URL.")
    parser.add_argument("--model", default="model.json", help="Model JSON file in sign_language_app/.")
    parser.add_argument("--k", type=int, default=9, help="KNN neighbor count.")
    parser.add_argument("--history-size", type=int, default=8, help="Frames kept for stable voting.")
    parser.add_argument("--stable-count", type=int, default=5, help="Votes required before a sign is stable.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Ignore model results below this confidence.")
    parser.add_argument("--no-srt", action="store_true", help="Do not generate .srt subtitle file.")
    parser.add_argument("--no-txt", action="store_true", help="Do not generate .txt timeline file.")
    parser.add_argument("--min-duration", type=float, default=0.2, help="Minimum duration of a gesture segment in seconds.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a GUI window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames.")
    parser.add_argument("--frame-interval", type=int, default=1, help="Process one frame every N frames to speed up.")
    return parser.parse_args()


def load_model(model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    with open(model_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["samples"], model_path


def get_stream_url(video_input):
    if not video_input.startswith(("http://", "https://")):
        return video_input

    options = {
        "format": "worst[ext=mp4]/worst",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        try:
            return ydl.extract_info(video_input, download=False)["url"]
        except Exception as exc:
            print(f"Cannot open YouTube video: {exc}")
            sys.exit(1)


def stable_from_history(history, stable_count):
    if not history:
        return "No hand", 0.0

    label, count = Counter(history).most_common(1)[0]
    confidence = count / len(history)
    if count < stable_count:
        return "No hand", confidence
    return label, confidence


def append_sentence(sentence, last_added_sign, stable_sign):
    if stable_sign == "No hand":
        return last_added_sign
    if stable_sign != "Unknown" and stable_sign != last_added_sign:
        sentence.append(stable_sign)
        return stable_sign
    return last_added_sign


def draw_sentence(frame, sentence):
    if not sentence:
        return

    height, width, _ = frame.shape
    text = " -> ".join(display_label(label, include_english=False) for label in sentence)
    max_chars = max(12, int(width / 16))
    if len(text) > max_chars:
        text = "..." + text[-max_chars:]
    draw_panel(
        frame,
        [{"text": f"句子: {text}", "color": (0, 255, 0), "font_size": 24}],
        origin=(10, height - 58),
        width=width - 20,
        row_height=34,
    )


def format_srt_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    if milliseconds > 999:
        milliseconds = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def save_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            start_str = format_srt_time(seg["start_time"])
            end_str = format_srt_time(seg["end_time"])
            label_text = display_label(seg["label"], include_english=False)
            f.write(f"{idx}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{label_text}\n\n")


def save_timeline_txt(segments, txt_path):
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("手語時間軸辨識結果\n")
        f.write("=" * 40 + "\n")
        for seg in segments:
            start_str = format_srt_time(seg["start_time"])
            end_str = format_srt_time(seg["end_time"])
            label_text = display_label(seg["label"], include_english=False)
            f.write(f"[{start_str} -> {end_str}] {label_text}\n")


def main():
    args = build_args()
    samples, model_path = load_model(args.model)
    video_source = get_stream_url(args.input)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Cannot open video: {args.input}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    history = deque(maxlen=args.history_size)
    sentence = []
    last_added_sign = None

    frame_idx = 0
    segments = []
    active_segment = None

    print(f"Loaded model: {model_path}")
    print("Press q to quit. Press c to clear the sentence.")
    if not args.headless:
        cv2.namedWindow("Sign Language Video Recognizer", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            current_time = frame_idx / fps
            if frame_idx % args.frame_interval != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb)

            current_sign = "No hand"
            confidence = 0.0

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                if not args.headless:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                query_vector = normalize_landmarks(mediapipe_landmarks_to_list(landmarks))
                current_sign, confidence = classify_knn(query_vector, samples, k=args.k)
                if confidence < args.min_confidence:
                    current_sign = "Unknown"

                if not args.headless:
                    x_vals = [landmark.x for landmark in landmarks.landmark]
                    y_vals = [landmark.y for landmark in landmarks.landmark]
                    x_pos = max(10, int(min(x_vals) * frame.shape[1]))
                    y_pos = max(30, int(min(y_vals) * frame.shape[0]) - 28)
                    draw_text(
                        frame,
                        f"{display_label(current_sign)} {confidence:.0%}",
                        (x_pos, y_pos),
                        22,
                        (255, 255, 0),
                    )
                history.append(current_sign)
            else:
                history.append("No hand")

            stable_sign, stable_confidence = stable_from_history(history, args.stable_count)
            last_added_sign = append_sentence(sentence, last_added_sign, stable_sign)

            # 時間軸片段追蹤邏輯
            current_label = None if stable_sign in ("No hand", "Unknown") else stable_sign
            active_label = active_segment["label"] if active_segment else None

            if current_label != active_label:
                if active_segment:
                    start_t = active_segment["start_time"]
                    end_t = current_time
                    if end_t - start_t >= args.min_duration:
                        segments.append({
                            "label": active_segment["label"],
                            "start_time": start_t,
                            "end_time": end_t
                        })
                    active_segment = None
                
                if current_label is not None:
                    active_segment = {
                        "label": current_label,
                        "start_time": current_time
                    }

            if not args.headless:
                if stable_sign != "No hand":
                    draw_panel(
                        frame,
                        [
                            {
                                "text": f"手語: {display_label(stable_sign)}  {stable_confidence:.0%}",
                                "color": (0, 255, 255),
                                "font_size": 24,
                            }
                        ],
                        width=460,
                    )

                draw_sentence(frame, sentence)
                cv2.imshow("Sign Language Video Recognizer", frame)

                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    sentence = []
                    last_added_sign = None
                if cv2.getWindowProperty("Sign Language Video Recognizer", cv2.WND_PROP_VISIBLE) < 1:
                    break

            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"Reached max frames: {args.max_frames}")
                break
    finally:
        # 影片結束時，結算最後一個手勢片段
        if active_segment:
            start_t = active_segment["start_time"]
            end_t = frame_idx / fps
            if end_t - start_t >= args.min_duration:
                segments.append({
                    "label": active_segment["label"],
                    "start_time": start_t,
                    "end_time": end_t
                })

        cap.release()
        hands_detector.close()
        if not args.headless:
            cv2.destroyAllWindows()

        print("\nRecognition result")
        print("=" * 40)
        if sentence:
            print(" -> ".join(display_label(label, include_english=False) for label in sentence))
        else:
            print("No stable sign was recognized.")
        print("=" * 40)

        # 輸出並儲存字幕檔與時間軸文字檔
        if segments:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            is_url = args.input.startswith(("http://", "https://"))
            if is_url:
                output_dir = script_dir
                base_name = "youtube_output"
            else:
                output_dir = os.path.dirname(os.path.abspath(args.input))
                base_name = os.path.splitext(os.path.basename(args.input))[0]

            print("\nTimeline result:")
            print("-" * 40)
            for seg in segments:
                start_s = format_srt_time(seg["start_time"])
                end_s = format_srt_time(seg["end_time"])
                lbl = display_label(seg["label"], include_english=False)
                print(f"[{start_s} -> {end_s}] {lbl}")
            print("-" * 40)

            if not args.no_srt:
                srt_path = os.path.join(output_dir, f"{base_name}.srt")
                save_srt(segments, srt_path)
                print(f"Subtitles saved to: {srt_path}")

            if not args.no_txt:
                txt_path = os.path.join(output_dir, f"{base_name}_timeline.txt")
                save_timeline_txt(segments, txt_path)
                print(f"Timeline transcript saved to: {txt_path}")


if __name__ == "__main__":
    main()
