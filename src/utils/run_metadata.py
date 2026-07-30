"""Run-metadata stamping: capture everything needed to reproduce a result.

The spec requires storing experiment configuration alongside every result.
Config alone isn't quite enough to explain a result months later, so this
also records the exact code version (git commit), library versions, and
when the run happened.
"""

import dataclasses
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union

import torch


def _get_git_commit_hash() -> str:
    """Return the current git commit hash, or "unknown" outside a git repo.

    Best-effort: metadata stamping should never fail a run just because git
    is unavailable (e.g. a fresh clone downloaded as a zip).
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def build_run_metadata(config: Any, device: torch.device) -> dict:
    """Build a JSON-serializable dict describing the environment and config
    a run was executed under.

    Args:
        config: the `ExperimentConfig` (or any dataclass) for this run.
        device: the `torch.device` the run executed on.

    Returns:
        A dict with timestamp, git commit, package versions, device, and
        the full config, ready to be saved next to the run's results.
    """
    config_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _get_git_commit_hash(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "config": config_dict,
    }


def save_run_metadata(metadata: dict, path: Union[str, Path]) -> None:
    """Write run metadata to disk as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
