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
    parser = argparse.ArgumentParser(description="Recognize trained sign-language labels from camera.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--model", default="model.json", help="Model JSON file in sign_language_app/.")
    parser.add_argument("--k", type=int, default=9, help="KNN neighbor count.")
    parser.add_argument("--history-size", type=int, default=8, help="Frames kept for stable voting.")
    parser.add_argument("--stable-count", type=int, default=5, help="Votes required before a sign is stable.")
    parser.add_argument("--min-confidence", type=float, default=0.45, help="Ignore model results below this confidence.")
    return parser.parse_args()


def load_model(model_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, model_name)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run downloader.py, extract_dataset.py, then train_classifier.py first.")
        sys.exit(1)

    with open(model_path, "r", encoding="utf-8") as handle:
        return json.load(handle)["samples"], model_path


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


def main():
    args = build_args()
    samples, model_path = load_model(args.model)
    print(f"Loaded model: {model_path}")
    print(f"Samples: {len(samples)}")
    print("Press q to quit. Press c to clear the sentence.")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    history = deque(maxlen=args.history_size)
    sentence = []
    last_added_sign = None

    cv2.namedWindow("Sign Language Recognition App", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Cannot read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(rgb)

            current_sign = "No hand"
            confidence = 0.0

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                query_vector = normalize_landmarks(mediapipe_landmarks_to_list(landmarks))
                current_sign, confidence = classify_knn(query_vector, samples, k=args.k)
                if confidence < args.min_confidence:
                    current_sign = "Unknown"

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
            cv2.imshow("Sign Language Recognition App", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                sentence = []
                last_added_sign = None
            if cv2.getWindowProperty("Sign Language Recognition App", cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        hands_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
