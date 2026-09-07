$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "Cannot find .venv. Create the Python 3.10 virtual environment first."
}

$code = @"
import ast
from pathlib import Path

for filename in (
    "main.py",
    "core/detectors/arm_detector.py",
    "core/detectors/face_detector.py",
    "core/detectors/hand_detector.py",
    "core/evaluation.py",
    "sign_language_app/labels.py",
    "tests/test_camera_scan.py",
    "tests/test_camera_mesh.py",
    "tests/test_hand_sign_regressions.py",
    "tests/mediapipe_hand_landmarks_v2.py",
):
    source = Path(filename).read_text(encoding="utf-8")
    ast.parse(source, filename=filename)

print("Syntax check passed.")
"@

& $python --version
Push-Location $projectRoot
try {
    $code | & $python -
}
finally {
    Pop-Location
}
