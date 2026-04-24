from __future__ import annotations

import argparse
import json
from pathlib import Path


def tail_lines(path: Path, lines: int) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(errors="replace").splitlines()[-lines:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-dir", required=True)
    parser.add_argument("--tail-lines", type=int, default=20)
    args = parser.parse_args()

    unit_dir = Path(args.unit_dir).resolve()
    status_path = unit_dir / "run_status.json"
    if not status_path.exists():
        raise SystemExit(f"no run_status.json under {unit_dir}")
    status = json.loads(status_path.read_text())
    payload = {"status": status}
    stdout_path = Path(status["stdout_path"])
    stderr_path = Path(status["stderr_path"])
    payload["stdout_tail"] = tail_lines(stdout_path, args.tail_lines)
    payload["stderr_tail"] = tail_lines(stderr_path, args.tail_lines)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
