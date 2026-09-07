"""快速掃描影片，統計偵測到的手部與伸直手指比例。"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.detectors.hand_detector import HandSignRecognizer


def analyze_video(video_path: str, sample_frames: int = 20) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"無法開啟影片：{video_path}")

    recognizer = HandSignRecognizer()
    total_hands = 0
    straight_fingers = 0
    frames_read = 0
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, total_frames // max(sample_frames, 1))
        while True:
            success, frame = cap.read()
            if not success:
                break
            if frames_read % interval == 0:
                detections = recognizer.process(frame)
                for detection in detections:
                    total_hands += 1
                    straight_fingers += sum(detection["fingers"].values())
            frames_read += 1
    finally:
        cap.release()
        recognizer.close()

    if frames_read == 0:
        raise ValueError(f"影片沒有可讀取的影格：{video_path}")
    return {
        "video": os.path.basename(video_path),
        "frames_read": frames_read,
        "hands": total_hands,
        "straight_fingers": straight_fingers,
        "straight_ratio": straight_fingers / max(total_hands * 5, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", help="要掃描的影片路徑")
    parser.add_argument("--sample-frames", type=int, default=20)
    args = parser.parse_args()

    failed = False
    for video_path in args.videos:
        try:
            result = analyze_video(video_path, args.sample_frames)
        except (OSError, ValueError) as exc:
            print(f"錯誤：{exc}", file=sys.stderr)
            failed = True
            continue
        print(
            f"{result['video']}: 讀取 {result['frames_read']} 幀，"
            f"偵測手部 {result['hands']} 次，"
            f"伸直手指比例 {result['straight_ratio']:.1%}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
