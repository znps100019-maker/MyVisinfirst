"""Deprecated: classification rules now live in core.detectors.hand_detector.

This file is intentionally read-only. Older versions rewrote source files with
string replacement and could reintroduce incorrect OK/7/8 logic.
"""

if __name__ == "__main__":
    raise SystemExit(
        "Deprecated script. Edit core/detectors/hand_detector.py and run "
        "python -m unittest discover -s tests -p 'test_*.py'."
    )
