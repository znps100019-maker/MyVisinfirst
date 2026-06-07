param(
    [Parameter(Mandatory = $true)]
    [string] $Title,

    [string] $Body = "Syntax check passed."
)

$ErrorActionPreference = "Stop"

& (Join-Path $PSScriptRoot "check_syntax.ps1")

function Resolve-Git {
    $systemGit = Get-Command git -ErrorAction SilentlyContinue
    if ($systemGit) {
        return $systemGit.Source
    }

    $localGit = Join-Path $PSScriptRoot "..\tools\mingit-2.54.0\cmd\git.exe"
    if (Test-Path $localGit) {
        return $localGit
    }

    throw "Cannot find git. Install Git before creating a PR."
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "Cannot find GitHub CLI (gh). Install and sign in before creating a PR."
}

$git = Resolve-Git

& $git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) {
    throw "This folder is not a Git repository yet. Run git init and add a GitHub remote before creating a PR."
}

$remote = & $git remote get-url origin 2>$null
if ($LASTEXITCODE -ne 0 -or -not $remote) {
    throw "No GitHub remote named origin was found. Add origin before creating a PR."
}

$branch = "codex/update-" + (Get-Date -Format "yyyyMMdd-HHmmss")
& $git switch -c $branch
& $git add .
& $git commit -m $Title
& $git push -u origin $branch
gh pr create --draft --title $Title --body $Body
