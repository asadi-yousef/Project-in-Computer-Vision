import pytest
import torch

from src.visualization.feature_space import (
    FeatureVisualizationSelection,
    load_selection,
    plot_feature_space,
    project_features_and_prototypes,
    save_selection,
    select_classes_and_samples,
)


def _synthetic_labels(num_classes=10, samples_per_class=20):
    return torch.tensor([c for c in range(num_classes) for _ in range(samples_per_class)])


def test_select_classes_and_samples_picks_requested_counts():
    labels = _synthetic_labels(num_classes=10, samples_per_class=20)

    selection = select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)

    assert len(selection.class_ids) == 5
    assert len(selection.sample_indices) == 20  # 5 classes * 4 samples
    assert len(set(selection.sample_indices)) == 20  # no duplicate indices

    selected_labels = labels[selection.sample_indices].tolist()
    assert set(selected_labels) == set(selection.class_ids)


def test_select_classes_and_samples_is_deterministic_given_same_seed():
    labels = _synthetic_labels()
    first = select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)
    second = select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)
    assert first == second


def test_select_classes_and_samples_raises_when_too_many_classes_requested():
    labels = _synthetic_labels(num_classes=3, samples_per_class=20)
    with pytest.raises(ValueError, match="only has 3"):
        select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)


def test_select_classes_and_samples_raises_when_class_too_small():
    labels = _synthetic_labels(num_classes=5, samples_per_class=2)
    with pytest.raises(ValueError, match="fewer than requested"):
        select_classes_and_samples("dtd", labels, num_classes=5, samples_per_class=4, seed=0)


def test_save_and_load_selection_round_trips(tmp_path):
    selection = FeatureVisualizationSelection(dataset="dtd", class_ids=[1, 3, 5], sample_indices=[10, 11, 12])
    path = tmp_path / "selection.json"

    save_selection(selection, path)
    loaded = load_selection(path)

    assert loaded == selection


def test_project_features_and_prototypes_returns_correct_shapes():
    torch.manual_seed(0)
    sample_features = torch.randn(30, 8)
    prototype_features = torch.randn(3, 8)

    sample_2d, prototype_2d = project_features_and_prototypes(sample_features, prototype_features, seed=0)

    assert sample_2d.shape == (30, 2)
    assert prototype_2d.shape == (3, 2)


def test_project_features_and_prototypes_normalizes_before_projecting():
    # Two orthogonal prototypes (unit norm by construction); "image"
    # features pointing in the same directions but with a huge norm -
    # reproducing the real scale mismatch (raw ResNet-18 features can have
    # norms ~50x a prototype's). Without normalizing samples first, t-SNE's
    # distance computation is dominated by norm rather than direction, and
    # both prototypes would collapse together regardless of class.
    torch.manual_seed(0)
    dim = 8
    prototype_features = torch.zeros(2, dim)
    prototype_features[0, 0] = 1.0
    prototype_features[1, 1] = 1.0

    noise = torch.randn(8, dim) * 0.01
    class_0_samples = prototype_features[0].unsqueeze(0).repeat(4, 1) * 50.0 + noise[:4]
    class_1_samples = prototype_features[1].unsqueeze(0).repeat(4, 1) * 50.0 + noise[4:]
    sample_features = torch.cat([class_0_samples, class_1_samples], dim=0)

    sample_2d, prototype_2d = project_features_and_prototypes(
        sample_features, prototype_features, seed=0
    )

    class_0_center = sample_2d[:4].mean(axis=0)
    class_1_center = sample_2d[4:].mean(axis=0)

    dist_0_to_proto0 = ((class_0_center - prototype_2d[0]) ** 2).sum()
    dist_0_to_proto1 = ((class_0_center - prototype_2d[1]) ** 2).sum()
    dist_1_to_proto0 = ((class_1_center - prototype_2d[0]) ** 2).sum()
    dist_1_to_proto1 = ((class_1_center - prototype_2d[1]) ** 2).sum()

    assert dist_0_to_proto0 < dist_0_to_proto1
    assert dist_1_to_proto1 < dist_1_to_proto0


def test_plot_feature_space_is_saved_to_disk(tmp_path):
    sample_2d = torch.randn(12, 2).numpy()
    sample_class_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    prototype_2d = torch.randn(3, 2).numpy()
    prototype_class_ids = [0, 1, 2]
    class_names = ["a", "b", "c"]

    save_path = tmp_path / "feature_space.png"
    plot_feature_space(
        sample_2d, sample_class_ids, prototype_2d, prototype_class_ids, class_names,
        "test plot", save_path,
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
