# MyVisinfirst

Arm movement detection project using OpenCV, MediaPipe, PyAutoGUI, and NumPy.
The current focus is hand-joint image recognition: MediaPipe detects 21 hand
landmarks, the program classifies simple static signs, and you can choose a
target sign to detect during your project demo.

## Structure

- `main.py` - starts the camera loop and triggers keyboard input.
- `arm_detector.py` - detects arm landmarks and calculates the elbow angle.
- `hand_detector.py` - detects MediaPipe hand joints and classifies simple static signs.
- `tests/` - folder containing camera testing and MediaPipe task scripts.
  - `test_camera_scan.py` - utility to scan for available webcam index numbers.
  - `test_camera_mesh.py` - real-time visualization of hand joint skeletons and mesh lines.
  - `mediapipe_hand_landmarks_v2.py` - new hand tracking using MediaPipe Tasks v2 HandLandmarker.
- `requirements.txt` - runtime dependencies.
- `scripts/check_syntax.ps1` - verifies Python syntax.
- `scripts/run_vision.ps1` - starts the camera app through an English drive path.
- `scripts/pr_after_check.ps1` - runs syntax checks, then opens a draft PR.
- `.github/workflows/syntax-check.yml` - GitHub Actions syntax check for pushes and PRs.

## Setup

This project expects Python 3.10.

This workspace has been prepared with:

- `.python310\python.exe` - local Python 3.10.11 installation.
- `.venv\Scripts\python.exe` - project virtual environment.

Use the existing environment:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip check
```

If you need to recreate it:

```powershell
.\.python310\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If Python 3.10 is already installed globally, you can use that instead:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To use the virtual environment in the current PowerShell session:

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

To detect a specific target sign:

```powershell
.\scripts\run_vision.ps1 --target-sign "Fist"
.\scripts\run_vision.ps1 --target-sign "OK"
.\scripts\run_vision.ps1 --target-sign "Number 1"
```

When the target is stable for several frames, the program prints a JSON event:

```json
{"event":"target_detected","target_sign":"Fist"}
```

To find the working camera index:

```powershell
.\.venv\Scripts\python.exe tests/test_camera_scan.py
```

To preview the hand mesh only:

```powershell
.\scripts\run_test.ps1
```

Useful options:

- `--camera 0` - choose the camera index.
- `--width 640 --height 480` - request a smaller camera frame for slower boards.
- `--headless` - run without `cv2.imshow`.
- `--disable-keyboard` - skip PyAutoGUI key presses.
- `--print-joints` - print detected arm and hand joints as JSON lines.
- `--max-frames 60` - stop automatically after a fixed number of frames.
- `--target-sign "Fist"` - mark a specific hand sign as the detection target.
- `--target-cooldown 1.0` - delay repeated target events.

## Current Sign Recognition

The first version uses MediaPipe Hands landmarks and rule-based finger states.
It currently recognizes:

- `Open palm`
- `Fist`
- `Thumbs up`
- `OK`
- `Number 1`
- `Number 2`
- `Number 3`
- `Number 4`
- `I love you`
- `Pinky`
- `Unknown`

This is a starter recognizer for static hand shapes. Full sign language
recognition should add trained examples, motion history, and sign-specific
labels.

## Check And PR

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_syntax.ps1
.\scripts\pr_after_check.ps1 -Title "Describe the change"
```
