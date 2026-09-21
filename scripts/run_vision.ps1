$ErrorActionPreference = "Stop"

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$driveLetter = $null

foreach ($candidate in @("M:", "N:", "O:", "P:", "Q:", "R:", "S:", "T:")) {
    $candidateRoot = "$candidate\"
    if (-not (Test-Path $candidateRoot)) {
        $driveLetter = $candidate
        break
    }

    if (Test-Path (Join-Path $candidateRoot "main.py")) {
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
} elseif (-not (Test-Path (Join-Path $mappedRoot "main.py"))) {
    throw "$driveLetter is already in use. Change the drive letter in scripts\run_vision.ps1."
}

$helper = Join-Path $PSScriptRoot "resolve_python.ps1"
. $helper
$python = Resolve-ProjectPython -ProjectRoot $mappedRoot
$main = Join-Path $mappedRoot "main.py"

& $python $main @args
