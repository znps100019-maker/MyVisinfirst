# Analysis tools

The old `improve_*`, `fix_*`, `restore_*`, and `optimize_*` files used string
replacement to rewrite source code. They are retained only as explicit
deprecation stubs and are not part of the normal run or test flow.

Classification changes belong in `core/detectors/hand_detector.py` and must be
covered by `tests/test_hand_sign_regressions.py`. Threshold changes require
human-labelled fixtures; reducing `Unknown` is not an accuracy measurement.
