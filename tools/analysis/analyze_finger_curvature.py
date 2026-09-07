"""Deprecated diagnostic entry point.

Finger-extension decisions now live in core.detectors.hand_detector and are
covered by the labelled regression tests. Use main.py or the maintained tests
for diagnostics.
"""
import sys


def main() -> int:
    print(
        "此診斷工具已棄用；請使用 "
        "python -m unittest discover -s tests -p 'test_*.py'"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
