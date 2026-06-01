$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonScripts = Join-Path $projectRoot ".venv\Scripts"
$gitCmd = Join-Path $projectRoot "tools\mingit-2.54.0\cmd"

if (Test-Path $pythonScripts) {
    $env:Path = "$pythonScripts;$env:Path"
}

if (Test-Path $gitCmd) {
    $env:Path = "$gitCmd;$env:Path"
}

python --version
if (Get-Command git -ErrorAction SilentlyContinue) {
    git --version
}
