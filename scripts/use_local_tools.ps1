$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$pythonScripts = Join-Path $projectRoot ".venv\Scripts"
$gitCmd = Join-Path $projectRoot "tools\mingit-2.54.0\cmd"

$env:Path = "$pythonScripts;$gitCmd;$env:Path"

python --version
git --version
