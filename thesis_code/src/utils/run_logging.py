from __future__ import annotations

import copy
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def create_run_directory(base_output_dir: str | os.PathLike[str], config_path: str | os.PathLike[str], run_name: str | None = None) -> dict[str, str]:
    """Create a timestamped run directory and copy relevant config files into it."""
    base_dir = Path(base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_name is None:
        run_name = f"run_{timestamp}"
    else:
        run_name = f"{run_name}_{timestamp}"

    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(config_path)
    config_copy_path = run_dir / config_path.name
    if config_path.exists():
        shutil.copy2(config_path, config_copy_path)

    metadata = {
        "run_dir": str(run_dir),
        "run_name": run_name,
        "timestamp": timestamp,
        "config_copy": str(config_copy_path),
    }
    return metadata


def save_config_snapshot(run_dir: str | os.PathLike[str], config: dict[str, Any]) -> str:
    """Persist the resolved config to YAML inside the run directory."""
    run_dir = Path(run_dir)
    snapshot_path = run_dir / "config_snapshot.yaml"
    with snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return str(snapshot_path)


def create_epoch_log(run_dir: str | os.PathLike[str], log_name: str = "training_log.txt") -> object:
    """Create a log file object for appending epoch details."""
    run_dir = Path(run_dir)
    log_path = run_dir / log_name
    log_path.touch(exist_ok=True)
    return log_path


def append_log_line(log_path: str | os.PathLike[str], message: str) -> None:
    """Append a single line to a log file."""
    with Path(log_path).open("a", encoding="utf-8") as f:
        f.write(message + "\n")
