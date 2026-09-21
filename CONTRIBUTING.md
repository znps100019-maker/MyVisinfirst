# Project Rule

Before every GitHub pull request:

1. Make the code change.
2. Run the syntax check:

   ```powershell
   .\scripts\check_syntax.ps1
   ```

   If Windows blocks unsigned local scripts, run the same check explicitly:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_syntax.ps1
   ```

3. Create a PR only after the syntax check passes:

   ```powershell
   .\scripts\pr_after_check.ps1 -Title "Describe the change"
   ```

The PR script creates a `codex/update-YYYYMMDD-HHMMSS` branch, commits the current
changes, pushes the branch, and opens a draft pull request.

If system Git is not installed, the PR script uses the local Git executable at
`tools\mingit-2.54.0\cmd\git.exe`.

## Python Environment

This project expects Python 3.10.

Recommended setup:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If the old `.venv` points to a removed Python installation, use
`.\scripts\resolve_python.ps1` for project checks and camera startup, or
recreate `.venv` with the commands above.

For board-style testing, prefer:

```powershell
.\scripts\run_vision.ps1 --headless --disable-keyboard --print-joints
```
