import os
import subprocess
import sys
import string

def handle_non_ascii_path():
    """Bypasses MediaPipe bug on Windows where non-ASCII characters in path cause it to fail loading binarypb files."""
    if sys.platform != "win32":
        return

    # Find project root - assuming utils/path_fix.py is 1 directory deep
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    # Check if the path actually contains non-ASCII characters
    if not any(ord(char) > 127 for char in project_root):
        return

    # Find a free drive letter
    drive = None
    for letter in string.ascii_uppercase[::-1]:
        candidate = f"{letter}:"
        if not os.path.exists(candidate + "\\"):
            drive = candidate
            break

    if not drive:
        print("Error: No free drive letter found to bypass MediaPipe path bug.")
        sys.exit(1)

    # Subst the drive to the project root
    subprocess.run(["subst", drive, project_root], shell=True, stdout=subprocess.DEVNULL)
    
    # Determine the script being executed
    # sys.argv[0] is the original script path (could be absolute or relative)
    original_script = os.path.abspath(sys.argv[0])
    try:
        relative_script = os.path.relpath(original_script, project_root)
    except ValueError:
        # Script is outside project root, might not need bypass or can't be bypassed this way
        relative_script = os.path.basename(original_script)
        
    virtual_script = os.path.join(drive, relative_script)
    
    # Check for virtual environment python
    virtual_python = os.path.join(drive, ".venv", "Scripts", "python.exe")
    if not os.path.exists(virtual_python):
        # Fallback to the executable that started the process, modifying its path if needed
        virtual_python = sys.executable.replace(project_root, drive)

    try:
        result = subprocess.run([virtual_python, virtual_script] + sys.argv[1:])
        returncode = result.returncode
    except Exception as e:
        print(f"Error during execution via subst drive: {e}")
        returncode = 1
    finally:
        subprocess.run(["subst", drive, "/d"], shell=True, stdout=subprocess.DEVNULL)

    sys.exit(returncode)
