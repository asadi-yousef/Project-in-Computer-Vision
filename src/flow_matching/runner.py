"""Orchestrates one full flow-matching experiment: load cached features,
restrict to the same K-shot subset Stage 1 used, build the same class
prototypes, train the velocity network, transport the test split, classify
by cosine similarity, and save everything alongside the config.

The single most important property of this module is that the subset, the
seeds, and the prototypes are *identical* to the corresponding Stage 1
prototype run - stage_2.pdf requires it, and the whole Stage 1 vs Stage 2
comparison is meaningless otherwise. That is why `prepare_features` calls
exactly the same `sample_balanced_subset_indices` and
`compute_class_prototypes` the Stage 1 runner calls, with the same
arguments, rather than reimplementing either.

This module is also the single place features are L2-normalized before the
flow. Stage 1's classifier already normalizes at both ends
(`compute_class_prototypes` normalizes its inputs, and
`predict_by_cosine_similarity` normalizes what it classifies), so putting
the flow in that same space is a matter of consistency, not a deviation -
and because cosine similarity is scale-invariant, it leaves the baseline
numbers untouched.
"""

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
import torch.nn.functional as F

from src.classifiers.prototype import compute_class_prototypes, predict_by_cosine_similarity
from src.data.few_shot import sample_balanced_subset_indices
from src.features.loading import load_validated_feature_cache
from src.flow_matching.inference import transport_with_checkpoint
from src.flow_matching.training import (
    FlowMatchingTrainResult,
    train_rolled_out_flow_matching,
    train_standard_flow_matching,
)
from src.utils.config import (
    FLOW_MATCHING_METHODS,
    VALID_EULER_STEPS,
    ExperimentConfig,
    save_config,
)
from src.utils.run_metadata import build_run_metadata, save_run_metadata


@dataclasses.dataclass
class PreparedFeatures:
    """Everything one FM run needs, prepared exactly as Stage 1 prepared it.

    `train_features`, `val_features` and `test_features` are L2-normalized
    (the flow operates on the unit sphere); `prototypes` are Stage 1's,
    unit-norm by construction.
    """

    train_features: torch.Tensor
    train_labels: torch.Tensor
    val_features: torch.Tensor
    val_labels: torch.Tensor
    test_features: torch.Tensor
    test_labels: torch.Tensor
    prototypes: torch.Tensor
    num_classes: int


def prepare_features(
    config: ExperimentConfig, cache_dir: Union[str, Path]
) -> PreparedFeatures:
    """Load cached features and reproduce Stage 1's subset and prototypes.

    Args:
        config: the run's configuration. For k_shot in {5, 10}, `config.seed`
            selects the balanced subset, exactly as in Stage 1; for
            k_shot="full" the whole official training split is used.
        cache_dir: directory holding cached features.

    Returns:
        A `PreparedFeatures` with normalized features and Stage 1's prototypes.
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
    num_classes = train_metadata["num_classes"]

    if config.k_shot != "full":
        indices = sample_balanced_subset_indices(
            train_labels.tolist(), config.k_shot, config.seed
        )
        train_features = train_features[indices]
        train_labels = train_labels[indices]

    # Computed from the raw subset features, exactly as the Stage 1 runner
    # does. (compute_class_prototypes normalizes internally, so this is
    # bit-identical either way - passing the raw features keeps the call site
    # literally the same as Stage 1's.)
    prototypes = compute_class_prototypes(train_features, train_labels, num_classes)

    return PreparedFeatures(
        train_features=F.normalize(train_features, dim=1),
        train_labels=train_labels,
        val_features=F.normalize(val_features, dim=1),
        val_labels=val_labels,
        test_features=F.normalize(test_features, dim=1),
        test_labels=test_labels,
        prototypes=prototypes,
        num_classes=num_classes,
    )


def flow_matching_run_dir(
    output_dir: Union[str, Path],
    dataset: str,
    encoder: str,
    method: str,
    k_shot,
    num_euler_steps: int,
    seed: int,
) -> Path:
    """Where one FM run's artifacts live.

    Mirrors Stage 1's layout with a T level inserted, and reuses the
    prototype branch's "single_run" convention for k_shot="full" - the FM
    full setting is one run, following the prototype baseline it is compared
    against (a project decision; see the run-count discussion in README).
    """
    seed_folder = f"seed{seed}" if k_shot != "full" else "single_run"
    return (
        Path(output_dir)
        / method
        / dataset
        / encoder
        / f"k{k_shot}"
        / f"T{num_euler_steps}"
        / seed_folder
    )


def evaluate_transported_accuracy(
    state_dict: Dict[str, torch.Tensor],
    hidden_dims: Sequence[int],
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    prototypes: torch.Tensor,
    num_euler_steps: int,
    device: torch.device,
) -> float:
    """Transport the test split and classify it against the class prototypes.

    Uses the same cosine-similarity rule as Stage 1
    (`predict_by_cosine_similarity`) against the same prototypes, so the only
    difference from the baseline is the transport itself.
    """
    transported = transport_with_checkpoint(
        state_dict, hidden_dims, test_features, num_euler_steps, device
    )
    predictions = predict_by_cosine_similarity(transported, prototypes)
    return (predictions == test_labels).float().mean().item()


def baseline_accuracy(
    test_features: torch.Tensor, test_labels: torch.Tensor, prototypes: torch.Tensor
) -> float:
    """The Stage 1 prototype accuracy for this exact setting.

    Recomputed here rather than read from Stage 1's outputs so every FM
    result carries its own baseline alongside it. That makes delta-accuracy
    self-contained, and doubles as an integrity check: if this ever stops
    matching the stored Stage 1 number, the subsets or prototypes have
    drifted and the comparison is invalid.
    """
    predictions = predict_by_cosine_similarity(test_features, prototypes)
    return (predictions == test_labels).float().mean().item()


def _save_run(
    config: ExperimentConfig,
    run_dir: Path,
    train_result: FlowMatchingTrainResult,
    result_fields: dict,
    device: torch.device,
) -> None:
    """Write config, history, result metadata, and checkpoint for one run."""
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, run_dir / "config.yaml")

    with open(run_dir / "history.json", "w") as f:
        json.dump([dataclasses.asdict(entry) for entry in train_result.history], f, indent=2)

    metadata = build_run_metadata(config, device)
    metadata["result"] = result_fields
    save_run_metadata(metadata, run_dir / "result.json")

    torch.save(train_result.final_state_dict, run_dir / "checkpoint.pt")


def _result_fields(
    test_accuracy: float,
    baseline_test_accuracy: float,
    num_euler_steps: int,
    train_result: FlowMatchingTrainResult,
) -> dict:
    return {
        "test_accuracy": test_accuracy,
        "baseline_test_accuracy": baseline_test_accuracy,
        "delta_accuracy": test_accuracy - baseline_test_accuracy,
        "num_euler_steps": num_euler_steps,
        "final_train_loss": train_result.history[-1].train_loss,
        "final_val_loss": train_result.history[-1].val_loss,
    }


def run_standard_fm_experiment(
    config: ExperimentConfig,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
    euler_step_counts: Sequence[int] = VALID_EULER_STEPS,
) -> List[dict]:
    """Train one standard-FM velocity network and evaluate it at several T.

    Standard FM's objective never discretizes the path, so the T=4 and T=12
    networks would be bit-identical under the same seed. Rather than train
    the same weights twice, this trains once and writes one self-contained
    run directory per T, each carrying its own config (recording that T) and
    a copy of the shared checkpoint.

    Args:
        config: must have method="fm_standard". Its
            `flow_matching.num_euler_steps` is ignored in favour of
            `euler_step_counts`.
        cache_dir: directory holding cached features.
        output_dir: root under which run directories are created.
        device: device to train and evaluate on.
        euler_step_counts: the T values to evaluate the trained network at.

    Returns:
        One result dict per T, each with test_accuracy, baseline_test_accuracy,
        delta_accuracy, num_euler_steps, and run_dir.
    """
    if config.method != "fm_standard":
        raise ValueError(f"config.method must be 'fm_standard', got {config.method!r}")

    data = prepare_features(config, cache_dir)
    train_result = train_standard_flow_matching(
        data.train_features,
        data.train_labels,
        data.prototypes,
        config.flow_matching,
        seed=config.seed,
        device=device,
        val_features=data.val_features,
        val_labels=data.val_labels,
    )

    baseline = baseline_accuracy(data.test_features, data.test_labels, data.prototypes)

    results = []
    for num_euler_steps in euler_step_counts:
        test_accuracy = evaluate_transported_accuracy(
            train_result.final_state_dict,
            config.flow_matching.hidden_dims,
            data.test_features,
            data.test_labels,
            data.prototypes,
            num_euler_steps,
            device,
        )
        # Each run directory records the T it was evaluated at, so the saved
        # config always describes the result sitting next to it.
        run_config = dataclasses.replace(
            config,
            flow_matching=dataclasses.replace(
                config.flow_matching, num_euler_steps=num_euler_steps
            ),
        )
        run_dir = flow_matching_run_dir(
            output_dir, config.dataset, config.encoder, config.method,
            config.k_shot, num_euler_steps, config.seed,
        )
        fields = _result_fields(test_accuracy, baseline, num_euler_steps, train_result)
        _save_run(run_config, run_dir, train_result, fields, device)
        results.append({**fields, "run_dir": str(run_dir)})

    return results


def run_rolled_out_fm_experiment(
    config: ExperimentConfig,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
) -> dict:
    """Train one rolled-out velocity network and evaluate it at its own T.

    Unlike standard FM, T is baked into the learned weights here, so training
    and evaluation must use the same T (stage_2.pdf) - one training per T,
    one run directory per training.

    Args:
        config: must have method="fm_rolled". `flow_matching.num_euler_steps`
            is used for both the training rollout and the evaluation.
        cache_dir: directory holding cached features.
        output_dir: root under which the run directory is created.
        device: device to train and evaluate on.

    Returns:
        A result dict with test_accuracy, baseline_test_accuracy,
        delta_accuracy, num_euler_steps, and run_dir.
    """
    if config.method != "fm_rolled":
        raise ValueError(f"config.method must be 'fm_rolled', got {config.method!r}")

    num_euler_steps = config.flow_matching.num_euler_steps

    data = prepare_features(config, cache_dir)
    train_result = train_rolled_out_flow_matching(
        data.train_features,
        data.train_labels,
        data.prototypes,
        config.flow_matching,
        seed=config.seed,
        device=device,
        val_features=data.val_features,
        val_labels=data.val_labels,
    )

    baseline = baseline_accuracy(data.test_features, data.test_labels, data.prototypes)
    test_accuracy = evaluate_transported_accuracy(
        train_result.final_state_dict,
        config.flow_matching.hidden_dims,
        data.test_features,
        data.test_labels,
        data.prototypes,
        num_euler_steps,
        device,
    )

    run_dir = flow_matching_run_dir(
        output_dir, config.dataset, config.encoder, config.method,
        config.k_shot, num_euler_steps, config.seed,
    )
    fields = _result_fields(test_accuracy, baseline, num_euler_steps, train_result)
    _save_run(config, run_dir, train_result, fields, device)

    return {**fields, "run_dir": str(run_dir)}


def run_flow_matching_experiment(
    config: ExperimentConfig,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    device: torch.device,
) -> List[dict]:
    """Dispatch to the right FM runner for `config.method`.

    Returns a list in both cases (standard FM produces one result per T,
    rolled-out exactly one) so callers can treat them uniformly.
    """
    if config.method == "fm_standard":
        return run_standard_fm_experiment(config, cache_dir, output_dir, device)
    if config.method == "fm_rolled":
        return [run_rolled_out_fm_experiment(config, cache_dir, output_dir, device)]
    raise ValueError(
        f"config.method must be one of {FLOW_MATCHING_METHODS}, got {config.method!r}"
    )
