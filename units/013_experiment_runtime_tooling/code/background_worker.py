from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_status(path: Path) -> dict:
    return json.loads(path.read_text())


def write_status(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    for _ in range(10):
        try:
            path.write_text(text)
            return
        except PermissionError:
            time.sleep(0.1)
    path.write_text(text)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: background_worker.py <status_path>")

    status_path = Path(sys.argv[1]).resolve()
    status = read_status(status_path)
    status["status"] = "running"
    status["worker_pid"] = None
    status["actual_start_utc"] = utc_now()
    write_status(status_path, status)

    stdout_path = Path(status["stdout_path"])
    stderr_path = Path(status["stderr_path"])
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = datetime.fromisoformat(status["scheduled_stop_utc"])
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    with stdout_path.open("ab") as stdout_handle, stderr_path.open("ab") as stderr_handle:
        try:
            proc = subprocess.Popen(
                status["command"],
                cwd=status["working_directory"],
                stdout=stdout_handle,
                stderr=stderr_handle,
                creationflags=creationflags,
            )
            status["python_pid"] = proc.pid
            write_status(status_path, status)

            timeout = max(1.0, deadline.timestamp() - time.time())
            try:
                return_code = proc.wait(timeout=timeout)
                status["exit_code"] = return_code
                status["status"] = "completed" if return_code == 0 else "failed"
                if return_code != 0:
                    status["stop_reason"] = "nonzero_exit"
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                status["status"] = "stopped"
                status["stop_reason"] = "deadline"
                status["timeout_hit"] = True
        except Exception as exc:  # noqa: BLE001
            status["status"] = "failed"
            status["stop_reason"] = "worker_exception"
            status["error"] = repr(exc)
        finally:
            status["end_utc"] = utc_now()
            write_status(status_path, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
