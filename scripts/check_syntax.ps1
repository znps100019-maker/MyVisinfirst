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
    "arm_detector.py",
    "hand_detector.py",
    "test.py",
    "test_camera_mesh.py",
    "mediapipe_hand_landmarks_01.py",
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
