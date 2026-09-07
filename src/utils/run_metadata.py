"""Run-metadata stamping and artifact writing: capture everything needed to
reproduce a result.

The spec requires storing experiment configuration alongside every result.
Config alone isn't quite enough to explain a result months later, so this
also records the exact code version (git commit), library versions, and
when the run happened.

`save_run_artifacts` owns the run-directory layout every stage shares -
config.yaml, history.json, result.json, checkpoint.pt - so the linear probe,
the Stage 2 flow-matching runs and the Stage 3 runs all produce directories
the same readers can walk.
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


def save_run_artifacts(
    config: Any,
    run_dir: Union[str, Path],
    state_dict: dict,
    history: list,
    result_fields: dict,
    device: torch.device,
) -> Path:
    """Write one run's four standard artifacts, creating the directory.

    Every runner in this project - linear probe, Stage 2 flow matching,
    Stage 3 - writes the same four files, and the aggregation and reporting
    code walks `outputs/` expecting exactly that. Keeping the layout in one
    place means a new stage cannot accidentally produce directories the
    report pipeline cannot read.

    Args:
        config: the run's `ExperimentConfig`, saved as config.yaml and
            embedded in result.json's metadata.
        run_dir: where to write. Created if absent.
        state_dict: model weights, saved as checkpoint.pt. Which weights
            these are is the caller's decision and differs by stage: Stage 1
            and Stage 3 save the best-validation checkpoint, Stage 2 the
            final one.
        history: per-epoch log dataclasses, saved as history.json.
        result_fields: the run's headline numbers, placed under
            result.json's "result" key.
        device: recorded in the metadata.

    Returns:
        The run directory.
    """
    # Imported here rather than at module scope: config.py is a heavier
    # import than this module's other dependencies, and every caller of the
    # metadata helpers above does not need it.
    from src.utils.config import save_config

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config.yaml")

    with open(run_dir / "history.json", "w") as f:
        json.dump([dataclasses.asdict(entry) for entry in history], f, indent=2)

    metadata = build_run_metadata(config, device)
    metadata["result"] = result_fields
    save_run_metadata(metadata, run_dir / "result.json")

    torch.save(state_dict, run_dir / "checkpoint.pt")

    return run_dir

