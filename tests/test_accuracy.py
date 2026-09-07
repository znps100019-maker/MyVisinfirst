"""Optional labelled-video evaluation CLI.

Without --label this reports coverage and distribution only. It never calls
coverage "accuracy".
"""
import argparse
import os

import cv2

from core.evaluation import evaluate_predictions


def evaluate_video(video_path, label=None, max_frames=0):
    from core.detectors.hand_detector import HandSignRecognizer

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    recognizer = HandSignRecognizer()
    predictions = []
    frames = 0
    try:
        while max_frames <= 0 or frames < max_frames:
            success, frame = cap.read()
            if not success:
                break
            detections = recognizer.process(frame)
            predictions.append(detections[0]["sign"] if detections else "No hand")
            frames += 1
    finally:
        cap.release()
        recognizer.close()

    if frames == 0:
        raise RuntimeError(f"Video contains zero readable frames: {video_path}")

    labels = [label] * len(predictions) if label else None
    return evaluate_predictions(predictions, labels), frames


def main():
    parser = argparse.ArgumentParser(description="Evaluate a labelled hand-shape video.")
    parser.add_argument("--video", required=True, help="Video path.")
    parser.add_argument("--label", default="", help="Expected label for every readable frame.")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    from utils.path_fix import handle_non_ascii_path

    handle_non_ascii_path()
    result, frames = evaluate_video(args.video, args.label or None, args.max_frames)
    print(f"影片: {os.path.basename(args.video)}")
    print(f"影格數: {frames}")
    print(f"覆蓋率: {result['coverage']:.1%}")
    print(f"分布: {result['distribution']}")
    if result["has_ground_truth"]:
        print(f"人工標註準確率: {result['accuracy']:.1%}")


if __name__ == "__main__":
    main()
