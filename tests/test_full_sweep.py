import torch

from src.features.cache import FeatureCacheMetadata, cache_file_path, save_feature_cache
from src.full_sweep import (
    flow_matching_result_path,
    linear_probe_result_path,
    prototype_result_path,
    run_full_sweep,
    seeds_for_k_shot,
)
from src.utils.config import FlowMatchingHyperparams


def _write_synthetic_cache(
    cache_dir, dataset, encoder, split, samples_per_class, num_classes, feature_dim, seed
):
    generator = torch.Generator().manual_seed(seed)
    features, labels = [], []
    for class_index in range(num_classes):
        center = torch.zeros(feature_dim)
        center[class_index % feature_dim] = 8.0
        noise = torch.randn(samples_per_class, feature_dim, generator=generator) * 0.1
        features.append(center.unsqueeze(0) + noise)
        labels.append(torch.full((samples_per_class,), class_index, dtype=torch.long))
    features_tensor, labels_tensor = torch.cat(features), torch.cat(labels)
    metadata = FeatureCacheMetadata(
        dataset=dataset, encoder=encoder, split=split,
        num_samples=features_tensor.shape[0], feature_dim=feature_dim, num_classes=num_classes,
    )
    save_feature_cache(
        cache_file_path(cache_dir, dataset, encoder, split), features_tensor, labels_tensor, metadata
    )


_TINY_FM = FlowMatchingHyperparams(
    hidden_dims=[16, 16], max_epochs=4, batch_size=16, learning_rate=1e-2
)


def _seed_synthetic_cache(cache_dir):
    for split, seed in [("train", 0), ("val", 1), ("test", 2)]:
        _write_synthetic_cache(
            cache_dir, "dtd", "resnet18", split,
            samples_per_class=20, num_classes=3, feature_dim=8, seed=seed,
        )


def test_run_full_sweep_produces_expected_result_files(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"  # unused: cache already present, so extraction is skipped
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        data_dir, cache_dir, output_dir, device=torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10, "full"], seeds=[0],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )

    assert linear_probe_result_path(output_dir, "dtd", "resnet18", 10, 0).exists()
    assert linear_probe_result_path(output_dir, "dtd", "resnet18", "full", 0).exists()
    assert prototype_result_path(output_dir, "dtd", "resnet18", 10, 0).exists()
    assert prototype_result_path(output_dir, "dtd", "resnet18", "full", 0).exists()


def test_run_full_sweep_skips_already_completed_runs(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )
    capsys.readouterr()  # discard first run's output

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )
    captured = capsys.readouterr()

    assert "already done, skipping" in captured.out
    assert "already cached, skipping" in captured.out


def test_run_full_sweep_respects_force_rerun(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    data_dir = tmp_path / "data"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )
    result_path = linear_probe_result_path(output_dir, "dtd", "resnet18", 10, 0)
    first_mtime = result_path.stat().st_mtime_ns

    run_full_sweep(
        data_dir, cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0], force_rerun=True,
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )
    second_mtime = result_path.stat().st_mtime_ns

    assert second_mtime >= first_mtime


# --- Stage 2: flow-matching sweep ---


def test_seeds_for_k_shot_repeats_few_shot_but_not_full():
    # The full setting is one run, following the Stage 1 prototype branch.
    assert seeds_for_k_shot(5, [0, 1, 2]) == [0, 1, 2]
    assert seeds_for_k_shot(10, [0, 1, 2]) == [0, 1, 2]
    assert seeds_for_k_shot("full", [0, 1, 2]) == [0]


def test_sweep_produces_every_flow_matching_run(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4, 12], flow_matching_hyperparams=_TINY_FM,
    )

    for method in ("fm_standard", "fm_rolled"):
        for num_euler_steps in (4, 12):
            path = flow_matching_result_path(
                output_dir, "dtd", "resnet18", method, 10, num_euler_steps, 0
            )
            assert path.exists(), f"missing {path}"


def test_full_setting_writes_one_run_not_three(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=["full"], seeds=[0, 1, 2],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )

    standard_runs = list((output_dir / "fm_standard").rglob("result.json"))
    rolled_runs = list((output_dir / "fm_rolled").rglob("result.json"))
    assert len(standard_runs) == 1
    assert len(rolled_runs) == 1
    assert "single_run" in str(standard_runs[0])


def test_few_shot_setting_writes_one_run_per_seed(tmp_path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0, 1, 2],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )

    assert len(list((output_dir / "fm_standard").rglob("result.json"))) == 3
    assert len(list((output_dir / "fm_rolled").rglob("result.json"))) == 3


def test_sweep_skips_completed_flow_matching_runs(tmp_path, capsys):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _seed_synthetic_cache(cache_dir)

    kwargs = dict(
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4, 12], flow_matching_hyperparams=_TINY_FM,
    )
    run_full_sweep(tmp_path / "data", cache_dir, output_dir, torch.device("cpu"), **kwargs)
    capsys.readouterr()

    run_full_sweep(tmp_path / "data", cache_dir, output_dir, torch.device("cpu"), **kwargs)
    captured = capsys.readouterr()

    assert "fm_standard dtd/resnet18 k=10 seed=0: already done, skipping" in captured.out
    assert "fm_rolled dtd/resnet18 k=10 T=4 seed=0: already done, skipping" in captured.out
    assert "fm_rolled dtd/resnet18 k=10 T=12 seed=0: already done, skipping" in captured.out


def test_standard_fm_setting_reruns_when_only_one_step_count_is_present(tmp_path, capsys):
    # Standard FM's T values share one training, so a half-finished setting
    # has no per-T training to resume - it must retrain rather than skip.
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )
    capsys.readouterr()

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4, 12], flow_matching_hyperparams=_TINY_FM,
    )
    captured = capsys.readouterr()

    assert "fm_standard dtd/resnet18 k=10 seed=0: running" in captured.out
    assert flow_matching_result_path(
        output_dir, "dtd", "resnet18", "fm_standard", 10, 12, 0
    ).exists()


def test_rolled_out_runs_are_skipped_independently_per_step_count(tmp_path, capsys):
    # Unlike standard FM, each T is its own training and resumes on its own.
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )
    capsys.readouterr()

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4, 12], flow_matching_hyperparams=_TINY_FM,
    )
    captured = capsys.readouterr()

    assert "fm_rolled dtd/resnet18 k=10 T=4 seed=0: already done, skipping" in captured.out
    assert "fm_rolled dtd/resnet18 k=10 T=12 seed=0: running" in captured.out


def test_flow_matching_runs_do_not_disturb_stage_1_outputs(tmp_path):
    # Stage 2 must add directories alongside Stage 1's, never inside them.
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "outputs"
    _seed_synthetic_cache(cache_dir)

    run_full_sweep(
        tmp_path / "data", cache_dir, output_dir, torch.device("cpu"),
        dataset_encoder_pairs=[("dtd", "resnet18")], k_shots=[10], seeds=[0],
        euler_step_counts=[4], flow_matching_hyperparams=_TINY_FM,
    )

    assert {p.name for p in output_dir.iterdir()} == {
        "linear_probe", "prototype", "fm_standard", "fm_rolled",
    }
    assert not list((output_dir / "prototype").rglob("checkpoint.pt"))
