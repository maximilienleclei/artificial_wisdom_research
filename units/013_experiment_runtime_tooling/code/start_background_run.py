from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def pick_worker_python(python_exe: str) -> str:
    path = Path(python_exe)
    if path.name.lower() == "python.exe":
        candidate = path.with_name("pythonw.exe")
        if candidate.exists():
            return str(candidate)
    return python_exe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--slice-seconds", required=True, type=int)
    parser.add_argument("--python-exe", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.command:
        raise SystemExit("background command is required")
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        raise SystemExit("background command is required")

    unit_dir = Path(args.unit_dir).resolve()
    model_run_dir = unit_dir / "model" / "runs" / args.run_name
    plot_run_dir = unit_dir / "plot" / "runs" / args.run_name
    model_run_dir.mkdir(parents=True, exist_ok=True)
    plot_run_dir.mkdir(parents=True, exist_ok=True)
    status_path = unit_dir / "run_status.json"

    status = {
        "unit_dir": str(unit_dir),
        "run_name": args.run_name,
        "status": "queued",
        "command": [args.python_exe, *command],
        "working_directory": str(unit_dir),
        "stdout_path": str(plot_run_dir / "stdout.log"),
        "stderr_path": str(plot_run_dir / "stderr.log"),
        "model_run_dir": str(model_run_dir),
        "plot_run_dir": str(plot_run_dir),
        "created_utc": utc_now(),
        "scheduled_stop_utc": (datetime.now(timezone.utc) + timedelta(seconds=args.slice_seconds)).isoformat(),
    }
    write_status(status_path, status)

    worker_script = Path(__file__).with_name("background_worker.py").resolve()
    worker_python = pick_worker_python(args.python_exe)
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        [worker_python, str(worker_script), str(status_path)],
        cwd=str(unit_dir),
        creationflags=creationflags,
        close_fds=True,
    )
    status["worker_launcher_pid"] = proc.pid
    status["worker_python"] = worker_python
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
