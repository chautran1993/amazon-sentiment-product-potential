"""Start FastAPI backend as a detached Windows process."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "backend_api.log"
ERR_PATH = PROJECT_ROOT / "backend_api.err.log"


def main() -> None:
    python_exe = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not python_exe.exists():
        python_exe = Path(sys.executable)

    command = [
        str(python_exe),
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    with LOG_PATH.open("a", encoding="utf-8") as stdout_file, ERR_PATH.open(
        "a", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            close_fds=True,
        )

    print(f"Started backend API with PID {process.pid}")
    print("Backend URL: http://127.0.0.1:8000")
    print(f"Logs: {LOG_PATH}")
    print(f"Errors: {ERR_PATH}")


if __name__ == "__main__":
    main()
