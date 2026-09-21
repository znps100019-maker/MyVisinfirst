function Resolve-ProjectPython {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot
    )

    $sitePackages = Join-Path $ProjectRoot ".venv\Lib\site-packages"
    $candidates = @(
        (Join-Path $ProjectRoot ".venv\Scripts\python.exe"),
        (Join-Path $ProjectRoot ".uv-python\cpython-3.10.20-windows-x86_64-none\python.exe")
    )

    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) {
            continue
        }

        $previousErrorAction = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & $candidate -c "import sys; print(sys.version_info[:2])" *> $null
        $probeExitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorAction
        if ($probeExitCode -eq 0) {
            if ($candidate -like "*\.uv-python\*") {
                if (Test-Path -LiteralPath $sitePackages) {
                    $env:PYTHONPATH = if ($env:PYTHONPATH) {
                        "$sitePackages;$env:PYTHONPATH"
                    } else {
                        $sitePackages
                    }
                }
            }
            return $candidate
        }
    }

    throw "No usable project Python was found. Create .venv or .uv-python first."
}
