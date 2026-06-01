$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$driveLetter = $null

foreach ($candidate in @("M:", "N:", "O:", "P:", "Q:", "R:", "S:", "T:")) {
    $candidateRoot = "$candidate\"
    if (-not (Test-Path $candidateRoot)) {
        $driveLetter = $candidate
        break
    }

    if (Test-Path (Join-Path $candidateRoot "test_camera_mesh.py")) {
        $driveLetter = $candidate
        break
    }
}

if (-not $driveLetter) {
    throw "No free drive letter was found for MediaPipe startup."
}

$mappedRoot = "$driveLetter\"

if (-not (Test-Path $mappedRoot)) {
    cmd /c "subst $driveLetter `"$projectRoot`""
} elseif (-not (Test-Path (Join-Path $mappedRoot "test_camera_mesh.py"))) {
    throw "$driveLetter is already in use. Change the drive letter in scripts\run_test.ps1."
}

$python = Join-Path $mappedRoot ".venv\Scripts\python.exe"
$main = Join-Path $mappedRoot "test_camera_mesh.py"

& $python $main @args
