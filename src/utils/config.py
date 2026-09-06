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
VALID_METHODS = (
    "linear_probe",
    "prototype",
    "fm_standard",
    "fm_rolled",
    "fm_cls_rolled",
    "fm_cls_guided",
)
VALID_K_SHOTS = (5, 10, "full")

# stage_2.pdf evaluates exactly these two Euler-step counts. Restricted for
# the same reason as the lists above: a stray T would produce a run that
# silently sits outside the comparison grid.
VALID_EULER_STEPS = (4, 12)

# The Stage 2 methods, both of which train a velocity network on top of the
# frozen features. Grouped here so callers can ask "is this a flow-matching
# run?" without repeating the pair in several places.
FLOW_MATCHING_METHODS = ("fm_standard", "fm_rolled")

# The Stage 3 methods (part_3.pdf), both of which train a velocity network
# whose output feeds a *frozen* Stage 1 linear probe:
#   - fm_cls_rolled: end-to-end rolled-out classification training;
#   - fm_cls_guided: classifier-guided targets with the standard FM loss.
# Grouped here for the same reason as FLOW_MATCHING_METHODS above.
STAGE3_METHODS = ("fm_cls_rolled", "fm_cls_guided")

# part_3.pdf: "Choose a single number of Euler steps T and use it throughout
# Stage 3." This is that choice. Stage 2 measured T in {4, 12} across 18
# settings and found T barely moved the result, so the cheaper value wins -
# and it is cheaper here in a way it was not in Stage 2, because Strategy 1
# backpropagates through the whole T-step rollout. The sweep passes this
# everywhere so no Stage 3 run can drift onto a different T.
STAGE3_NUM_EULER_STEPS = 4

# Strategy 1's displacement-penalty weight. part_3.pdf offers this
# regularization as an option ("penalizing the displacement between z and
# z_hat"); this value was chosen on validation accuracy over {0, 0.1, 1, 10}
# at two learning rates and three seeds, with test accuracy held out.
#
# It is not cosmetic. Stage 1's probe already reaches 100% accuracy and a
# cross-entropy of about 0.002 on the very K-shot subset Stage 3 trains on,
# so the classification objective starts at its floor and the cheapest
# remaining descent direction is to inflate feature magnitude along the logit
# direction. Unregularized, validation accuracy peaks at epoch 1-4 and then
# falls below the untrained identity; at 0.1 it improves for 50-190 epochs.
# The unregularized setting is kept as a reported ablation.
STAGE3_DISPLACEMENT_PENALTY = 0.1

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
class Stage3Hyperparams:
    """Velocity-network and training configuration for Stage 3 (part_3.pdf).

    Stage 3 reuses Stage 2's velocity-network architecture and Euler
    integration, so the architecture and optimizer fields below deliberately
    mirror `FlowMatchingHyperparams`. This is a separate dataclass rather than
    extra fields on that one for two reasons: the Stage-3-only knobs are
    meaningless for Stage 2, and every Stage 2 run has already written a
    `flow_matching` block to disk that must keep round-tripping unchanged.

    Two departures from Stage 2 are load-bearing rather than cosmetic:

      - The flow runs in the *raw* feature space. Stage 2 L2-normalized
        because its classifier was cosine similarity, which is scale-invariant,
        so normalizing cost nothing. Stage 3's classifier is Stage 1's linear
        probe, fitted to unnormalized features with mean norms of roughly 24
        (ResNet-18) and 48 (DINOv2). Normalizing here would hand the frozen
        classifier a distribution it has never seen.

      - The network starts with zero output, so the complete pipeline
        reproduces the Stage 1 linear probe exactly before training
        (part_3.pdf: "Initialize the FM close to identity"). Stage 2 used
        PyTorch's default initialization, which moves DINOv2 features by about
        6% of their norm before a single update.

    Of Strategy 1's two regularization knobs, `velocity_penalty` defaults to 0
    and `displacement_penalty` does not; see STAGE3_DISPLACEMENT_PENALTY for
    the measurements behind that. Both address the same failure: with a frozen
    classifier the cheapest way to reduce cross-entropy is to inflate the
    feature magnitude along the logit direction, which is a scale hack rather
    than a better representation.

    Strategy 2's target knobs are the four values part_3.pdf explicitly leaves
    open. `target_step_size` is measured in units of the training subset's mean
    feature norm rather than as an absolute distance, so a single value means
    the same thing on both encoders despite their feature scales differing by
    roughly a factor of two.
    """

    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [512, 512])
    num_euler_steps: int = STAGE3_NUM_EULER_STEPS
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 200

    # Strategy 1 (fm_cls_rolled) only. 0 disables the penalty entirely; the
    # displacement default is the validation-selected value documented at
    # STAGE3_DISPLACEMENT_PENALTY, and the ablation overrides it to 0.
    displacement_penalty: float = STAGE3_DISPLACEMENT_PENALTY
    velocity_penalty: float = 0.0

    # Strategy 2 (fm_cls_guided) only: how the classifier-guided target
    # z_hat' is constructed, and how often it is refreshed during training.
    target_step_size: float = 0.1
    target_num_steps: int = 1
    target_refresh_epochs: int = 1
    normalize_target_update: bool = True

    def __post_init__(self) -> None:
        if self.num_euler_steps < 1:
            raise ValueError(
                f"num_euler_steps must be at least 1, got {self.num_euler_steps!r}"
            )
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one hidden layer width")
        if any(width <= 0 for width in self.hidden_dims):
            raise ValueError(
                f"hidden_dims widths must all be positive, got {self.hidden_dims!r}"
            )
        if self.displacement_penalty < 0:
            raise ValueError(
                f"displacement_penalty must be non-negative, "
                f"got {self.displacement_penalty}"
            )
        if self.velocity_penalty < 0:
            raise ValueError(
                f"velocity_penalty must be non-negative, got {self.velocity_penalty}"
            )
        if self.target_step_size <= 0:
            raise ValueError(
                f"target_step_size must be positive, got {self.target_step_size}"
            )
        if self.target_num_steps < 1:
            raise ValueError(
                f"target_num_steps must be at least 1, got {self.target_num_steps}"
            )
        if self.target_refresh_epochs < 1:
            raise ValueError(
                f"target_refresh_epochs must be at least 1, "
                f"got {self.target_refresh_epochs}"
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
    stage3: Stage3Hyperparams = dataclasses.field(default_factory=Stage3Hyperparams)

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

    Blocks introduced by a later stage are optional on read: configs saved
    before Stage 2 have no `flow_matching` block, and those saved before
    Stage 3 have no `stage3` block. Both load fine and pick up the
    defaults, so every already-saved run config stays readable.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    paths_raw = raw.pop("paths", {})
    linear_probe_raw = raw.pop("linear_probe", {})
    flow_matching_raw = raw.pop("flow_matching", {})
    stage3_raw = raw.pop("stage3", {})
    return ExperimentConfig(
        **raw,
        paths=Paths(**paths_raw),
        linear_probe=LinearProbeHyperparams(**linear_probe_raw),
        flow_matching=FlowMatchingHyperparams(**flow_matching_raw),
        stage3=Stage3Hyperparams(**stage3_raw),
    )


def save_config(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """Save an `ExperimentConfig` to a YAML file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)
