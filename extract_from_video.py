"""
從影片抽取手語特徵（切片平均版）
每段影片切成 N 個時間片段，每段平均成 1 個向量
檔名格式：標籤_編號.mp4
執行：python extract_from_video.py --dir sign_language_app\raw_videos
"""
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
FRAME_INTERVAL = 3
SLICES_PER_VIDEO = 4  # 每段影片切成幾個片段


def label_from_filename(filename):
    name = os.path.splitext(filename)[0]
    return re.sub(r"_\d+$", "", name)


def extract_one_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法開啟：{video_path}")
        return []

    hands = mp_hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.7, min_tracking_confidence=0.6,
    )
    face = mp_face.FaceMesh(
        static_image_mode=False, max_num_faces=1,
        min_detection_confidence=0.5,
    )

    # 收集所有幀的特徵
    all_vectors = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % FRAME_INTERVAL != 0:
            continue

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

    if not all_vectors:
        print(f"{os.path.basename(video_path)} → 0 筆（無手部）")
        return []

    # 切成 N 個片段，每段平均
    n = len(all_vectors)
    samples = []
    slice_size = max(1, n // SLICES_PER_VIDEO)
    for i in range(0, n, slice_size):
        chunk = all_vectors[i:i + slice_size]
        if not chunk:
            continue
        # 對每個維度取平均
        avg_vector = [
            sum(v[d] for v in chunk) / len(chunk)
            for d in range(len(chunk[0]))
        ]
        samples.append({"label": label, "vector": avg_vector})

    print(f"{os.path.basename(video_path)} → {len(all_vectors)} 幀 → {len(samples)} 筆（{label}）")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="sign_language_app/raw_videos")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"資料夾不存在：{args.dir}")
        return

    all_samples = []
    video_exts = (".mp4", ".avi", ".mov", ".mkv")
    for filename in sorted(os.listdir(args.dir)):
        if not filename.lower().endswith(video_exts):
            continue
        video_path = os.path.join(args.dir, filename)
        label = label_from_filename(filename)
        all_samples.extend(extract_one_video(video_path, label))

    if not all_samples:
        print("沒有抽到任何樣本")
        return

    from collections import Counter
    counter = Counter(s["label"] for s in all_samples)
    print(f"\n各標籤樣本數：{dict(counter)}")
    print(f"總樣本數：{len(all_samples)}")
    print(f"向量長度：{len(all_samples[0]['vector'])}")

    if os.path.exists(MODEL_PATH):
        backup = MODEL_PATH + ".before_extract"
        os.replace(MODEL_PATH, backup)
        print(f"舊檔已備份到 {backup}")

    with open(MODEL_PATH, "w", encoding="utf-8") as f:
        json.dump({"samples": all_samples}, f, ensure_ascii=False)
    print(f"已儲存到 {MODEL_PATH}")


if __name__ == "__main__":
    main()