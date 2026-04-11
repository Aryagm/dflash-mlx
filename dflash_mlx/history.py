from __future__ import annotations

import csv
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY_PATH = REPO_ROOT / "benchmarks" / "metrics_history.csv"


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_metadata() -> dict[str, Any]:
    branch = _git_output(["rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git_output(["rev-parse", "HEAD"])
    short_commit = _git_output(["rev-parse", "--short", "HEAD"])
    status = _git_output(["status", "--short"])
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_short_commit": short_commit,
        "git_dirty": bool(status),
    }


def run_metadata(script_name: str, experiment_tag: str | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": script_name,
        "experiment_tag": experiment_tag or "",
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        **git_metadata(),
    }


def _normalize_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=isinstance(value, dict))
    return str(value)


def append_rows(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        return

    normalized_rows = [{key: _normalize_value(value) for key, value in row.items()} for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: list[dict[str, str]] = []
    fieldnames: list[str] = []

    if path.exists():
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)

    merged_fields = list(fieldnames)
    for row in normalized_rows:
        for key in row:
            if key not in merged_fields:
                merged_fields.append(key)

    write_mode = "a" if path.exists() and merged_fields == fieldnames else "w"
    with path.open(write_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields, lineterminator="\n")
        if write_mode == "w":
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({key: row.get(key, "") for key in merged_fields})
        elif path.stat().st_size == 0:
            writer.writeheader()

        for row in normalized_rows:
            writer.writerow({key: row.get(key, "") for key in merged_fields})
