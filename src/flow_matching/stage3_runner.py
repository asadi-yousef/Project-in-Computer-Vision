"""Orchestrates one Stage 3 experiment: FM before a frozen linear classifier.

Loads cached features, restricts them to the same K-shot subset Stage 1
sampled, loads and verifies that seed's frozen Stage 1 linear probe, trains
the velocity network with one of part_3.pdf's two strategies, evaluates the
complete pipeline on the test split, and saves the run's artifacts.

The single most important property of this module is that the subset, the
seed, and the classifier are *identical* to the corresponding Stage 1
linear-probe run. part_3.pdf requires it ("Use the same data splits and
sampled training subsets as in the Stage 1 linear-probe experiments"), and
every reported delta is a paired comparison against that specific run's own
published number. `prepare_stage3_features` therefore calls the same
`sample_balanced_subset_indices` with the same arguments Stage 1 used, and
`load_frozen_linear_probe` reads that run's actual checkpoint rather than
retraining an equivalent one.

Unlike `src.flow_matching.runner`, this module does **not** L2-normalize.
Stage 2 normalized because its classifier was cosine similarity, which is
scale-invariant; Stage 1's linear probe was fitted to raw features with mean
norms of roughly 24 (ResNet-18) and 48 (DINOv2), and normalizing here costs
2.5-4 accuracy points before training even starts.
`verify_stored_test_accuracy` fails the run if this is ever violated.

The steps are exposed separately - `prepare_stage3_features`,
`train_stage3_method`, `evaluate_stage3_checkpoint` - rather than only as one
end-to-end call, because the hyperparameter search needs all of them but
none of the artifact writing: it evaluates hundreds of configurations and
must not leave hundreds of run directories behind.
"""

import dataclasses
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from src.classifiers.frozen_probe import (
    FrozenLinearProbe,
    load_frozen_linear_probe,
    verify_stored_test_accuracy,
)
from src.data.few_shot import sample_balanced_subset_indices
from src.features.loading import load_validated_feature_cache
from src.flow_matching.runner import flow_matching_run_dir
from src.flow_matching.stage3_training import (
    Stage3EpochLog,
    Stage3TrainResult,
    evaluate_pipeline,
    train_classifier_guided_fm,
    train_rolled_out_classification,
)
from src.flow_matching.velocity_net import build_near_identity_velocity_network
from src.utils.config import STAGE3_METHODS, ExperimentConfig
from src.utils.run_metadata import save_run_artifacts

# Which training function implements each of part_3.pdf's strategies.
STRATEGY_TRAINERS = {
    "fm_cls_rolled": train_rolled_out_classification,
    "fm_cls_guided": train_classifier_guided_fm,
}


@dataclasses.dataclass
class Stage3Data:
    """The splits one Stage 3 run needs, prepared exactly as Stage 1 had them.

    All features are **raw**: not L2-normalized, unlike Stage 2's
    `PreparedFeatures`. See the module docstring.

    `train_features`/`train_labels` are already restricted to the K-shot
    subset; the validation and test splits are always complete.
    """

    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    num_classes: int

    @property
    def feature_dim(self) -> int:
        return self.train_features.shape[1]


def prepare_stage3_features(
    config: ExperimentConfig, cache_dir: Union[str, Path]
) -> Stage3Data:
    """Load cached features and reproduce Stage 1's K-shot subset.

    Args:
        config: the run's configuration. For k_shot in {5, 10}, `config.seed`
            selects the balanced subset, exactly as in Stage 1; for
            k_shot="full" the whole official training split is used.
        cache_dir: directory holding cached features.

    Returns:
        A `Stage3Data` with raw, unnormalized features.
    """
    train_features, train_labels, train_metadata = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "train"
    )
    val_features, val_labels, _ = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "val"
    )
    test_features, test_labels, _ = load_validated_feature_cache(
        cache_dir, config.dataset, config.encoder, "test"
    )

    if config.k_shot != "full":
        indices = sample_balanced_subset_indices(
            train_labels.tolist(), config.k_shot, config.seed
        )
        train_features = train_features[indices]
        train_labels = train_labels[indices]

    return Stage3Data(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        num_classes=train_metadata["num_classes"],
    )


def load_stage3_classifier(
    config: ExperimentConfig,
    data: Stage3Data,
    output_dir: Union[str, Path],
    device: torch.device,
) -> FrozenLinearProbe:
    """Load this seed's Stage 1 probe, verify it, and freeze it.

    The verification is not optional bookkeeping. It is what makes the whole
    stage's arithmetic trustworthy: it re-scores the loaded checkpoint on the
    test split and refuses to continue unless it reproduces the accuracy
    Stage 1 published, catching a wrong seed, a rebuilt feature cache, or
    features prepared differently here than they were then.

    Args:
        config: identifies the Stage 1 run to pair with.
        data: prepared splits, used for the verification pass.
        output_dir: the project outputs root.
        device: device to place the classifier on.

    Returns:
        The frozen probe, carrying its stored Stage 1 test accuracy.

    Raises:
        ValueError: if the checkpoint no longer reproduces its stored accuracy.
        FileNotFoundError: if Stage 1 has not been run for this setting.
    """
    frozen = load_frozen_linear_probe(
        output_dir, config.dataset, config.encoder, config.k_shot, config.seed, device
    )
    verify_stored_test_accuracy(frozen, data.test_features, data.test_labels, device)
    return frozen


def train_stage3_method(
    method: str,
    data: Stage3Data,
    classifier: nn.Module,
    hyperparams,
    seed: int,
    device: torch.device,
    progress: Optional[Callable[[Stage3EpochLog], None]] = None,
) -> Stage3TrainResult:
    """Dispatch to the training function for one of part_3.pdf's strategies.

    Args:
        method: "fm_cls_rolled" or "fm_cls_guided".
        data: prepared splits.
        classifier: the frozen Stage 1 probe, on `device`.
        hyperparams: a `Stage3Hyperparams`.
        seed: drives initialization and shuffling.
        device: device to train on.
        progress: optional per-epoch callback.

    Returns:
        The `Stage3TrainResult` from the chosen strategy.

    Raises:
        ValueError: if `method` is not a Stage 3 method.
    """
    if method not in STRATEGY_TRAINERS:
        raise ValueError(f"method must be one of {STAGE3_METHODS}, got {method!r}")

    return STRATEGY_TRAINERS[method](
        data.train_features,
        data.train_labels,
        data.val_features,
        data.val_labels,
        classifier,
        data.num_classes,
        hyperparams,
        seed=seed,
        device=device,
        progress=progress,
    )


def evaluate_stage3_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    hidden_dims: Sequence[int],
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Rebuild a trained velocity network from weights and score the pipeline.

    Args:
        state_dict: weights saved by a Stage 3 training run.
        hidden_dims: the widths the network was built with; must match.
        classifier: the frozen Stage 1 probe, on `device`.
        features, labels: the split to score, raw and unnormalized.
        num_steps: T.
        device: device to evaluate on.

    Returns:
        (loss, accuracy, mean_displacement).
    """
    velocity_net = build_near_identity_velocity_network(
        features.shape[1], hidden_dims
    ).to(device)
    velocity_net.load_state_dict(state_dict)
    return evaluate_pipeline(
        velocity_net, classifier, features, labels, num_steps, device
    )


def stage3_result_fields(
    train_result: Stage3TrainResult,
    test_accuracy: float,
    test_displacement: float,
    baseline_test_accuracy: float,
    num_euler_steps: int,
) -> dict:
    """The headline numbers saved in a Stage 3 run's result.json.

    `test_accuracy`, `baseline_test_accuracy`, `delta_accuracy` and
    `num_euler_steps` deliberately match the keys Stage 2 writes, so
    `src.evaluation.aggregation` reads Stage 3 runs with no changes.

    The remaining fields are Stage 3's own. `initial_val_accuracy` is the
    untrained pipeline's accuracy - identical to the frozen probe's, since
    the flow starts as the exact identity - and is what makes it possible to
    say whether training helped at all. `test_mean_displacement` records how
    far the flow actually moved the features, in raw feature units, which is
    the diagnostic for a flow that improves its objective by inflating
    magnitude rather than by improving the representation.
    """
    return {
        "test_accuracy": test_accuracy,
        "baseline_test_accuracy": baseline_test_accuracy,
        "delta_accuracy": test_accuracy - baseline_test_accuracy,
        "num_euler_steps": num_euler_steps,
        "best_epoch": train_result.best_epoch,
        "best_val_accuracy": train_result.best_val_accuracy,
        "initial_val_accuracy": train_result.initial_val_accuracy,
        "val_delta_accuracy": (
            train_result.best_val_accuracy - train_result.initial_val_accuracy
        ),
        "test_mean_displacement": test_displacement,
        "final_train_loss": train_result.history[-1].train_loss,
        "final_val_loss": train_result.history[-1].val_loss,
    }


def stage3_run_dir(
    output_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    method: str,
    k_shot,
    num_euler_steps: int,
    seed: int,
) -> Path:
    """Where one Stage 3 run's artifacts live.

    Reuses Stage 2's layout helper unchanged - it is already parameterized by
    method - so Stage 3 runs sit alongside Stage 2's under `outputs/` in the
    same shape, and the report pipeline walks them identically.
    """
    return flow_matching_run_dir(
        output_dir, dataset, encoder, method, k_shot, num_euler_steps, seed
    )


def run_stage3_experiment(
    config: ExperimentConfig,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    progress: Optional[Callable[[Stage3EpochLog], None]] = None,
) -> dict:
    """Run one Stage 3 experiment end to end and save its artifacts.

    Args:
        config: must have a Stage 3 method. `config.stage3` supplies the
            architecture, optimizer and strategy settings.
        cache_dir: directory holding cached features.
        output_dir: the project outputs root. Both the Stage 1 checkpoint
            this run pairs with and this run's own directory are located
            beneath it.
        device: device to train and evaluate on.
        progress: optional per-epoch callback.

    Returns:
        The result fields, plus "run_dir".

    Raises:
        ValueError: if config.method is not a Stage 3 method, or the frozen
            classifier fails verification.
    """
    if config.method not in STAGE3_METHODS:
        raise ValueError(
            f"config.method must be one of {STAGE3_METHODS}, got {config.method!r}"
        )

    hyperparams = config.stage3
    data = prepare_stage3_features(config, cache_dir)
    frozen = load_stage3_classifier(config, data, output_dir, device)

    train_result = train_stage3_method(
        config.method, data, frozen.model, hyperparams, config.seed, device, progress
    )

    _, test_accuracy, test_displacement = evaluate_stage3_checkpoint(
        train_result.best_state_dict,
        hyperparams.hidden_dims,
        frozen.model,
        data.test_features,
        data.test_labels,
        hyperparams.num_euler_steps,
        device,
    )

    fields = stage3_result_fields(
        train_result,
        test_accuracy,
        test_displacement,
        frozen.stored_test_accuracy,
        hyperparams.num_euler_steps,
    )
    run_dir = stage3_run_dir(
        output_dir,
        config.dataset,
        config.encoder,
        config.method,
        config.k_shot,
        hyperparams.num_euler_steps,
        config.seed,
    )
    save_run_artifacts(
        config, run_dir, train_result.best_state_dict, train_result.history, fields, device
    )

    return {**fields, "run_dir": str(run_dir)}
