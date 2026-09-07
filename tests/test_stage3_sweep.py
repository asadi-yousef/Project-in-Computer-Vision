"""Tests for the Stage 3 sweep and the selected-hyperparameter table."""

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from src.classifiers.linear_probe import LinearProbe
from src.classifiers.linear_probe_runner import linear_probe_run_dir
from src.features.cache import cache_file_path
from src.stage3_sweep import run_stage3_sweep, stage3_result_path
from src.utils.config import (
    STAGE3_K_SHOT,
    STAGE3_METHODS,
    STAGE3_SEEDS,
    STAGE3_SELECTED_HYPERPARAMS,
    STAGE3_SETTINGS,
    Stage3Hyperparams,
    VALID_DATASETS,
    stage3_hyperparams_for,
)
from src.utils.seeding import set_seed

CPU = torch.device("cpu")
FEATURE_DIM = 12
NUM_CLASSES = 4
# The sweep is keyed by dataset, so the synthetic project must use real
# dataset names; these are the encoders STAGE3_SETTINGS pairs them with.
TEST_DATASETS = list(STAGE3_SETTINGS)


# --- the selected-hyperparameter table ---


def test_a_selection_exists_for_every_method_and_dataset():
    expected = {
        (method, dataset) for method in STAGE3_METHODS for dataset in STAGE3_SETTINGS
    }
    assert set(STAGE3_SELECTED_HYPERPARAMS) == expected


def test_every_selection_is_a_valid_override():
    for (method, dataset) in STAGE3_SELECTED_HYPERPARAMS:
        hyperparams = stage3_hyperparams_for(method, dataset)
        assert isinstance(hyperparams, Stage3Hyperparams)


def test_selections_override_only_the_searched_knobs():
    # Everything the search held fixed must come from the dataclass defaults,
    # so the two strategies stay comparable on architecture, optimizer,
    # epoch budget and T (part_3.pdf and stage_2.pdf both require this).
    defaults = Stage3Hyperparams()
    searched = {
        "displacement_penalty", "velocity_penalty",
        "target_step_size", "target_num_steps", "target_refresh_epochs",
    }

    for (method, dataset) in STAGE3_SELECTED_HYPERPARAMS:
        selected = stage3_hyperparams_for(method, dataset)
        for field in ("hidden_dims", "num_euler_steps", "learning_rate",
                      "weight_decay", "batch_size", "max_epochs", "optimizer",
                      "normalize_target_update"):
            assert getattr(selected, field) == getattr(defaults, field), field
        assert set(STAGE3_SELECTED_HYPERPARAMS[(method, dataset)]) <= searched


def test_a_missing_selection_raises_a_useful_error():
    with pytest.raises(KeyError, match="tune_stage3"):
        stage3_hyperparams_for("fm_cls_rolled", "flowers102_typo")


def test_selections_can_be_applied_to_a_custom_base():
    base = Stage3Hyperparams(max_epochs=3, hidden_dims=[8, 8])

    selected = stage3_hyperparams_for("fm_cls_guided", "dtd", base)

    assert selected.max_epochs == 3  # base is respected
    assert selected.hidden_dims == [8, 8]
    assert selected.target_step_size == 0.02  # selection is applied


def test_the_stage3_protocol_constants_match_the_spec():
    # part_3.pdf's narrowed scope: one encoder per dataset, one training-set
    # size ("a reasonable default is K = 10"), three repetitions per Stage 1.
    assert STAGE3_K_SHOT == 10
    assert STAGE3_SEEDS == [0, 1, 2]
    assert set(STAGE3_SETTINGS) <= set(VALID_DATASETS)
    assert STAGE3_SETTINGS["flowers102"] == "resnet18"


# --- the sweep ---


def _write_cache(cache_dir, dataset, encoder, split, num_per_class, seed):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.arange(NUM_CLASSES).repeat_interleave(num_per_class)
    features = torch.randn(len(labels), FEATURE_DIM, generator=generator) * 24.0
    features += F.one_hot(labels, FEATURE_DIM).float() * 24.0

    path = cache_file_path(cache_dir, dataset, encoder, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features, "labels": labels,
            "metadata": {
                "dataset": dataset, "encoder": encoder, "split": split,
                "num_samples": len(labels), "feature_dim": FEATURE_DIM,
                "num_classes": NUM_CLASSES,
            },
        },
        path,
    )


def _write_stage1_run(output_dir, cache_dir, dataset, encoder, k_shot, seed):
    set_seed(seed)
    probe = LinearProbe(FEATURE_DIM, NUM_CLASSES)
    cache = torch.load(cache_file_path(cache_dir, dataset, encoder, "test"), weights_only=False)
    with torch.no_grad():
        accuracy = (
            probe(cache["features"]).argmax(dim=1) == cache["labels"]
        ).float().mean().item()

    run_dir = linear_probe_run_dir(output_dir, dataset, encoder, k_shot, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), run_dir / "checkpoint.pt")
    with open(run_dir / "result.json", "w") as f:
        json.dump({"result": {"test_accuracy": accuracy}}, f)


@pytest.fixture
def project(tmp_path):
    """A synthetic project with caches and Stage 1 runs for both settings."""
    cache_dir, output_dir = tmp_path / "cache", tmp_path / "outputs"
    for dataset in TEST_DATASETS:
        encoder = STAGE3_SETTINGS[dataset]
        _write_cache(cache_dir, dataset, encoder, "train", 6, seed=1)
        _write_cache(cache_dir, dataset, encoder, "val", 4, seed=2)
        _write_cache(cache_dir, dataset, encoder, "test", 4, seed=3)
        for seed in (0, 1):
            _write_stage1_run(output_dir, cache_dir, dataset, encoder, 5, seed)
    return cache_dir, output_dir


TINY = Stage3Hyperparams(hidden_dims=[8, 8], num_euler_steps=2, max_epochs=2, batch_size=8)


def _sweep(cache_dir, output_dir, **kwargs):
    settings = dict(
        force_rerun=False, seeds=[0], k_shot=5, base_hyperparams=TINY, verbose=False
    )
    settings.update(kwargs)
    return run_stage3_sweep(cache_dir, output_dir, CPU, **settings)


def test_the_sweep_covers_every_method_and_dataset(project):
    cache_dir, output_dir = project

    results = _sweep(cache_dir, output_dir)

    assert len(results) == len(STAGE3_METHODS) * len(TEST_DATASETS)
    assert {(r["method"], r["dataset"]) for r in results} == {
        (m, d) for m in STAGE3_METHODS for d in TEST_DATASETS
    }


def test_every_seed_produces_its_own_run(project):
    cache_dir, output_dir = project

    results = _sweep(cache_dir, output_dir, seeds=[0, 1], methods=["fm_cls_rolled"],
                     datasets=["dtd"])

    assert sorted(r["seed"] for r in results) == [0, 1]
    assert len({r["run_dir"] for r in results}) == 2


def test_completed_runs_are_skipped(project):
    cache_dir, output_dir = project

    first = _sweep(cache_dir, output_dir, methods=["fm_cls_rolled"], datasets=["dtd"])
    second = _sweep(cache_dir, output_dir, methods=["fm_cls_rolled"], datasets=["dtd"])

    assert len(first) == 1
    assert second == []


def test_force_rerun_overrides_the_skip(project):
    cache_dir, output_dir = project

    _sweep(cache_dir, output_dir, methods=["fm_cls_rolled"], datasets=["dtd"])
    rerun = _sweep(cache_dir, output_dir, methods=["fm_cls_rolled"], datasets=["dtd"],
                   force_rerun=True)

    assert len(rerun) == 1


def test_each_run_writes_its_artifacts_where_the_result_path_says(project):
    cache_dir, output_dir = project

    results = _sweep(cache_dir, output_dir, methods=["fm_cls_guided"], datasets=["dtd"])

    expected = stage3_result_path(
        output_dir, "dtd", STAGE3_SETTINGS["dtd"], "fm_cls_guided", 5, 2, 0
    )
    assert expected.exists()
    assert Path(results[0]["run_dir"]) == expected.parent


def test_runs_use_their_datasets_selected_hyperparameters(project):
    from src.utils.config import load_config

    cache_dir, output_dir = project

    _sweep(cache_dir, output_dir, methods=["fm_cls_guided"], datasets=["flowers102"])

    run_dir = stage3_result_path(
        output_dir, "flowers102", "resnet18", "fm_cls_guided", 5, 2, 0
    ).parent
    saved = load_config(run_dir / "config.yaml").stage3

    selected = STAGE3_SELECTED_HYPERPARAMS[("fm_cls_guided", "flowers102")]
    assert saved.target_step_size == selected["target_step_size"]
    assert saved.target_num_steps == selected["target_num_steps"]
    assert saved.target_refresh_epochs == selected["target_refresh_epochs"]


def test_results_carry_the_paired_stage_1_baseline(project):
    cache_dir, output_dir = project

    results = _sweep(cache_dir, output_dir, methods=["fm_cls_rolled"], datasets=["dtd"])

    stage1_result = json.loads(
        (linear_probe_run_dir(output_dir, "dtd", "dinov2_vits14", 5, 0) / "result.json").read_text()
    )
    assert results[0]["baseline_test_accuracy"] == pytest.approx(
        stage1_result["result"]["test_accuracy"]
    )


def test_a_missing_stage_1_run_fails_loudly(project):
    cache_dir, output_dir = project

    with pytest.raises(FileNotFoundError, match="No Stage 1 checkpoint"):
        _sweep(cache_dir, output_dir, seeds=[2], methods=["fm_cls_rolled"],
               datasets=["dtd"])
