$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$helper = Join-Path $PSScriptRoot "resolve_python.ps1"
. $helper
$python = Resolve-ProjectPython -ProjectRoot $projectRoot

$code = @"
import ast
from pathlib import Path

for filename in (
    "main.py",
    "run_web.py",
    "core/detectors/arm_detector.py",
    "core/detectors/face_detector.py",
    "core/detectors/hand_detector.py",
    "core/evaluation.py",
    "sign_language_app/labels.py",
    "tools/download_expression_dataset.py",
    "tools/analysis/analyze_results.py",
    "tests/test_camera_scan.py",
    "tests/test_camera_mesh.py",
    "tests/test_face_detector.py",
    "tests/test_hand_sign_regressions.py",
    "tests/test_web_server.py",
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
