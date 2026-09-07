# Analysis tools

The old `improve_*`, `fix_*`, `restore_*`, `optimize_*`, and legacy diagnostic
files used string replacement or stale APIs. They are retained only as
explicit deprecation stubs and are not part of the normal run or test flow.

`quick_video_scan.py` and `create_dashboard.py` are the maintained optional
diagnostics. Both require an explicit video path; they do not depend on a
backup-video directory.

Classification changes belong in `core/detectors/hand_detector.py` and must be
covered by `tests/test_hand_sign_regressions.py`. Threshold changes require
human-labelled fixtures; reducing `Unknown` is not an accuracy measurement.
