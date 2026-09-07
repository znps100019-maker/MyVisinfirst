"""Deprecated diagnostic entry point.

Use the core recognizer and labelled regression tests; this old script used
the removed root-level hand_detector module.
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
