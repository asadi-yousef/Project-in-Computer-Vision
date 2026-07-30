"""Typed configuration schema for Stage 1 experiments, backed by YAML files.

Every experiment (feature extraction, linear-probe training, prototype
evaluation) is described by an `ExperimentConfig`. Keeping the schema in one
place means every script loads/saves configs the same way, and invalid
combinations (e.g. DINOv2 on the wrong dataset) are caught immediately
instead of silently producing a mismatched cache or checkpoint.
"""

import dataclasses
from pathlib import Path
from typing import Union

import yaml

# Only the datasets/encoders/methods actually selected for this project.
# Restricted on purpose: the spec allows a 3rd dataset and a CLIP branch,
# but this project never uses them, so allowing them here would let a typo
# silently pass validation instead of failing loudly.
VALID_DATASETS = ("dtd", "flowers102")
VALID_ENCODERS = ("resnet18", "dinov2_vits14")
VALID_METHODS = ("linear_probe", "prototype")
VALID_K_SHOTS = (5, 10, "full")

# DINOv2 was only selected for DTD (Task 0 decision), not Flowers-102.
DINOV2_DATASET = "dtd"


@dataclasses.dataclass
class Paths:
    """Where raw datasets, cached features, and outputs live on disk.

    Stored as plain strings (not `pathlib.Path`) so the dataclass can be
    dumped straight to YAML/JSON without a custom encoder; callers that need
    a `Path` should wrap the field themselves, e.g. `Path(config.paths.data_dir)`.
    """

    data_dir: str = "data"
    cache_dir: str = "cache"
    output_dir: str = "outputs"


@dataclasses.dataclass
class LinearProbeHyperparams:
    """Suggested linear-probe training configuration from stage_1.pdf.

    These are defaults, not fixed requirements: the spec explicitly allows
    adjusting them if validation results show they behave poorly.
    """

    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200


@dataclasses.dataclass
class ExperimentConfig:
    """Full description of a single Stage 1 experiment run.

    `seed` has a dual role by design (see project notes): for k_shot in
    {5, 10} it selects both the balanced training subset and the
    classifier's initialization/training stochasticity; for k_shot="full"
    there is no subset to select, so it only drives classifier
    initialization.
    """

    dataset: str
    encoder: str
    method: str
    k_shot: Union[int, str]
    seed: int
    paths: Paths = dataclasses.field(default_factory=Paths)
    linear_probe: LinearProbeHyperparams = dataclasses.field(
        default_factory=LinearProbeHyperparams
    )

    def __post_init__(self) -> None:
        if self.dataset not in VALID_DATASETS:
            raise ValueError(
                f"dataset must be one of {VALID_DATASETS}, got {self.dataset!r}"
            )
        if self.encoder not in VALID_ENCODERS:
            raise ValueError(
                f"encoder must be one of {VALID_ENCODERS}, got {self.encoder!r}"
            )
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS}, got {self.method!r}"
            )
        if self.k_shot not in VALID_K_SHOTS:
            raise ValueError(
                f"k_shot must be one of {VALID_K_SHOTS}, got {self.k_shot!r}"
            )
        if self.encoder == "dinov2_vits14" and self.dataset != DINOV2_DATASET:
            raise ValueError(
                "dinov2_vits14 is only used on "
                f"{DINOV2_DATASET!r} in this project, got dataset={self.dataset!r}"
            )
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """Load an `ExperimentConfig` from a YAML file."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    paths_raw = raw.pop("paths", {})
    linear_probe_raw = raw.pop("linear_probe", {})
    return ExperimentConfig(
        **raw,
        paths=Paths(**paths_raw),
        linear_probe=LinearProbeHyperparams(**linear_probe_raw),
    )


def save_config(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """Save an `ExperimentConfig` to a YAML file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)
