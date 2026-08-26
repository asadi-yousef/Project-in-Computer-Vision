"""Typed configuration schema for this project's experiments, backed by YAML files.

Every experiment (feature extraction, linear-probe training, prototype
evaluation, flow-matching training) is described by an `ExperimentConfig`.
Keeping the schema in one place means every script loads/saves configs the
same way, and invalid combinations (e.g. DINOv2 on the wrong dataset) are
caught immediately instead of silently producing a mismatched cache or
checkpoint.
"""

import dataclasses
from pathlib import Path
from typing import List, Union

import yaml

# Only the datasets/encoders/methods actually selected for this project.
# Restricted on purpose: the spec allows a 3rd dataset and a CLIP branch,
# but this project never uses them, so allowing them here would let a typo
# silently pass validation instead of failing loudly.
VALID_DATASETS = ("dtd", "flowers102")
VALID_ENCODERS = ("resnet18", "dinov2_vits14")
VALID_METHODS = ("linear_probe", "prototype", "fm_standard", "fm_rolled")
VALID_K_SHOTS = (5, 10, "full")

# stage_2.pdf evaluates exactly these two Euler-step counts. Restricted for
# the same reason as the lists above: a stray T would produce a run that
# silently sits outside the comparison grid.
VALID_EULER_STEPS = (4, 12)

# The Stage 2 methods, both of which train a velocity network on top of the
# frozen features. Grouped here so callers can ask "is this a flow-matching
# run?" without repeating the pair in several places.
FLOW_MATCHING_METHODS = ("fm_standard", "fm_rolled")

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
class FlowMatchingHyperparams:
    """Velocity-network architecture and training configuration (stage_2.pdf).

    Architecture defaults follow the spec's suggestion: a small MLP with two
    hidden layers of width ~512 and SiLU activations. The optimizer defaults
    deliberately mirror `LinearProbeHyperparams` rather than being tuned
    separately - stage_2.pdf explicitly says no extensive hyperparameter
    search is needed, and reusing Stage 1's settings keeps the comparison
    between the two stages clean.

    `num_euler_steps` (T) means different things to the two methods, which is
    why it lives here rather than being purely a training or inference knob:
      - fm_standard: inference only. The training objective samples t
        continuously and never discretizes the path, so one trained network
        serves every T.
      - fm_rolled: training *and* inference. The rollout is unrolled T steps
        while training, so T is baked into the learned weights and must match
        at inference (stage_2.pdf: "use the same value of T during training
        and inference").

    `hidden_dims` is a list rather than a tuple so the config round-trips
    through YAML: `yaml.safe_dump` writes a tuple with a Python-specific tag
    that `yaml.safe_load` then refuses to read back.
    """

    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [512, 512])
    num_euler_steps: int = 4
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200

    def __post_init__(self) -> None:
        if self.num_euler_steps not in VALID_EULER_STEPS:
            raise ValueError(
                f"num_euler_steps must be one of {VALID_EULER_STEPS}, "
                f"got {self.num_euler_steps!r}"
            )
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer width")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError(
                f"hidden_dims widths must all be positive, got {self.hidden_dims!r}"
            )


@dataclasses.dataclass
class ExperimentConfig:
    """Full description of a single experiment run.

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
    flow_matching: FlowMatchingHyperparams = dataclasses.field(
        default_factory=FlowMatchingHyperparams
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
    """Load an `ExperimentConfig` from a YAML file.

    Configs saved before Stage 2 have no `flow_matching` block; they load
    fine and pick up the defaults, so Stage 1's already-saved run configs
    stay readable.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    paths_raw = raw.pop("paths", {})
    linear_probe_raw = raw.pop("linear_probe", {})
    flow_matching_raw = raw.pop("flow_matching", {})
    return ExperimentConfig(
        **raw,
        paths=Paths(**paths_raw),
        linear_probe=LinearProbeHyperparams(**linear_probe_raw),
        flow_matching=FlowMatchingHyperparams(**flow_matching_raw),
    )


def save_config(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """Save an `ExperimentConfig` to a YAML file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)
