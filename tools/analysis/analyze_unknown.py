"""Deprecated diagnostic entry point.

This script depended on removed thumb-spread APIs. Unknown handling is now
validated through labelled regression tests instead of threshold-only reports.
"""
import sys


def main() -> int:
    print(
        "此診斷工具已棄用；Unknown 需使用人工標註資料評估，"
        "不能以 Unknown 比例當作準確率。"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
