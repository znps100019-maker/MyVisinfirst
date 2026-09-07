"""Deprecated diagnostic entry point.

The former script duplicated loose finger thresholds. The maintained
implementation is core.detectors.hand_detector.HandSignRecognizer.
"""
import sys


def main() -> int:
    print(
        "此診斷工具已棄用；請直接使用 core.detectors.hand_detector "
        "與標註回歸測試。"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
