# MyVisinfirst

Arm movement detection project using OpenCV, MediaPipe, PyAutoGUI, and NumPy.

## Structure

- `main.py` - starts the camera loop and triggers keyboard input.
- `arm_detector.py` - detects arm landmarks and calculates the elbow angle.
- `hand_detector.py` - detects MediaPipe hand joints and classifies simple static signs.
- `requirements.txt` - runtime dependencies.
- `scripts/check_syntax.ps1` - verifies Python syntax.
- `scripts/run_vision.ps1` - starts the camera app through an English drive path.
- `scripts/pr_after_check.ps1` - runs syntax checks, then opens a draft PR.
- `.github/workflows/syntax-check.yml` - GitHub Actions syntax check for pushes and PRs.

## Setup

This project expects Python 3.10.

```powershell
uv venv --python 3.10 --seed .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This workspace already has a local Python 3.10 virtual environment at `.venv`.
It also has local Git at `tools\mingit-2.54.0\cmd\git.exe`.

To use both in the current PowerShell session:

```powershell
. .\scripts\use_local_tools.ps1
```

## Run

```powershell
.\scripts\run_vision.ps1
```

The run script maps this project to a temporary `M:` drive before starting
Python. This avoids a MediaPipe model-loading issue when the project path
contains non-English characters.

For board-style testing without a display window:

```powershell
.\scripts\run_vision.ps1 --headless --disable-keyboard --print-joints
```

Useful options:

- `--camera 0` - choose the camera index.
- `--width 640 --height 480` - request a smaller camera frame for slower boards.
- `--headless` - run without `cv2.imshow`.
- `--disable-keyboard` - skip PyAutoGUI key presses.
- `--print-joints` - print detected arm and hand joints as JSON lines.

## Current Sign Recognition

The first version uses MediaPipe Hands landmarks and rule-based finger states.
It currently recognizes:

- `Open palm`
- `Fist`
- `Thumbs up`
- `V / Peace`
- `Point / 1`
- `I love you`
- `Pinky`
- `Unknown`

This is a starter recognizer for static hand shapes. Full sign language
recognition should add trained examples, motion history, and sign-specific
labels.

## Check And PR

```powershell
.\scripts\check_syntax.ps1
.\scripts\pr_after_check.ps1 -Title "Describe the change"
```
