import argparse
import cv2
import mediapipe as mp
import json
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sign_language_app"))
from landmarks import (
    mediapipe_landmarks_to_list,
    mediapipe_face_to_list,
    build_full_vector,
)

mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh

MODEL_PATH = os.path.join("sign_language_app", "model.json")
FRAME_INTERVAL = 3 # 每 3 幀抽一次

def label_from_filename(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r"_\d+$", "", name)

def extract_one_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    all_vectors = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        if frame_idx % FRAME_INTERVAL != 0: continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        hand_results = hands.process(rgb)
        face_results = face.process(rgb)

        hands_list = []
        if hand_results.multi_hand_landmarks:
            for hand in hand_results.multi_hand_landmarks:
                hands_list.append(mediapipe_landmarks_to_list(hand))

        face_list = None
        if face_results.multi_face_landmarks:
            face_list = mediapipe_face_to_list(face_results.multi_face_landmarks[0])

        if hands_list:
            vector = build_full_vector(hands_list, face_list)
            all_vectors.append(vector)

    cap.release()
    hands.close()
    face.close()

    if not all_vectors: return []

    # 【關鍵】不再算軌跡，直接算整段影片的「平均特徵」
    n = len(all_vectors)
    avg_vector = [
        sum(v[d] for v in all_vectors) / n
        for d in range(len(all_vectors[0]))
    ]
    
    print(f"{os.path.basename(video_path)} → 平均 {n} 幀 → 1 筆（{label}）")
    return [{"label": label, "vector": avg_vector}]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="sign_language_app/raw_videos")
    args = parser.parse_args()

    all_samples = []
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    for filename in sorted(os.listdir(args.dir)):
        if not filename.lower().endswith(video_exts): continue
        video_path = os.path.join(args.dir, filename)
        label = label_from_filename(filename)
        all_samples.extend(extract_one_video(video_path, label))

    if not all_samples: return

    from collections import Counter
    counter = Counter(s["label"] for s in all_samples)
    print(f"\n總樣本數：{len(all_samples)}")
    
    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({"samples": all_samples}, f, ensure_ascii=False)
    print(f"已儲存到 {MODEL_PATH}")

if __name__ == "__main__":
    main()