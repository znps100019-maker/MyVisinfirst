"""Batch labelled-video evaluation CLI.

Every video must have a matching subdirectory name as its expected label.
Missing directories, unreadable videos, and zero-frame videos fail loudly.
"""
import argparse
from pathlib import Path

from tests.test_accuracy import evaluate_video


def main():
    parser = argparse.ArgumentParser(description="Evaluate all labelled videos in a directory.")
    parser.add_argument("--video-dir", required=True, help="Directory containing label subdirectories.")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.video_dir)
    if not root.is_dir():
        raise RuntimeError(f"Video directory does not exist: {root}")
    videos = sorted(root.rglob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No .mp4 videos found in: {root}")

    total_correct = 0
    total_frames = 0
    for video in videos:
        label = video.parent.name
        result, frames = evaluate_video(str(video), label, args.max_frames)
        total_correct += round(result["accuracy"] * frames)
        total_frames += frames
        print(f"{video.name}: {result['accuracy']:.1%} ({frames} frames)")

    print(f"Overall labelled accuracy: {total_correct / total_frames:.1%}")


if __name__ == "__main__":
    main()
