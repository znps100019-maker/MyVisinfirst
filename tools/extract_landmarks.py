"""Deprecated compatibility entry point.

Use sign_language_app/extract_dataset.py for the maintained dataset pipeline.
The old version imported a removed root-level hand_detector module and is kept
out of the normal workflow to avoid generating incompatible datasets.
"""
import sys


def main() -> int:
    print(
        "此工具已棄用，請改用："
        " python sign_language_app/extract_dataset.py"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
