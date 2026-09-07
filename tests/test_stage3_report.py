"""Tests for the Stage 3 report figures.

The generic helper changes (optional prototypes, optional normalization) are
tested here alongside the Stage 3 assembly that motivated them, with
integration tests against the real completed runs at the bottom.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.evaluation.stage3_report import (
    METHOD_PANEL_TITLES,
    Stage3Figures,
    _tuning_tables,
    class_separation,
    format_class_separation_table,
    format_stage3_observations,
    format_stage3_section,
    load_stage3_run,
    plot_stage3_feature_space,
    plot_stage3_training_curves,
    stage3_run_directory,
    summarize_stage3_outcomes,
)
from src.utils.config import STAGE3_K_SHOT, STAGE3_METHODS, STAGE3_SETTINGS
from src.visualization.feature_space import (
    plot_feature_space_comparison,
    project_feature_groups,
)

CPU = torch.device("cpu")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_OUTPUTS = PROJECT_ROOT / "outputs"
REAL_CACHE = PROJECT_ROOT / "cache"
REAL_REPORTS = PROJECT_ROOT / "reports"


# --- project_feature_groups: the normalize flag ---


def _corresponding_point_gap(projected):
    """Mean distance between matched points of two groups, as a fraction of
    the embedding's own spread.

    Compared as a ratio because t-SNE coordinates have no fixed scale, and
    because identical inputs land near each other but not on top of each
    other - it is a stochastic optimization, not a deterministic map.
    """
    first, second = projected
    gap = np.linalg.norm(first - second, axis=1).mean()
    spread = np.concatenate(projected).std()
    return gap / spread


def test_projection_normalizes_by_default():
    # Stage 2's behaviour must be unchanged: two groups differing only by a
    # scale factor are the same points on the unit sphere, so they land in
    # the same place.
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(40, 8, generator=generator)
    groups = [features * 10.0, features]

    normalized_gap = _corresponding_point_gap(project_feature_groups(groups, seed=0))
    raw_gap = _corresponding_point_gap(
        project_feature_groups(groups, seed=0, normalize=False)
    )

    assert normalized_gap < 0.25
    assert raw_gap > 2 * normalized_gap


def test_projection_can_keep_magnitudes():
    # Stage 3 needs this: its classifier is not scale-invariant, and the
    # displacement magnitude is what separates its two strategies. With
    # magnitudes kept, a 10x-scaled copy must land somewhere else entirely.
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(40, 8, generator=generator)

    gap = _corresponding_point_gap(
        project_feature_groups([features * 10.0, features], seed=0, normalize=False)
    )

    assert gap > 0.5


def test_projection_splits_groups_in_order_either_way():
    groups = [torch.randn(5, 4), torch.randn(7, 4), torch.randn(3, 4)]

    for normalize in (True, False):
        projected = project_feature_groups(groups, seed=0, normalize=normalize)
        assert [p.shape for p in projected] == [(5, 2), (7, 2), (3, 2)]


# --- plot_feature_space_comparison: optional prototypes ---


def _panels(count=3, num_samples=20):
    generator = np.random.default_rng(0)
    return [
        (f"panel {i}", generator.normal(size=(num_samples, 2))) for i in range(count)
    ]


def test_the_comparison_plots_without_prototypes(tmp_path):
    path = tmp_path / "no_prototypes.png"

    plot_feature_space_comparison(
        _panels(),
        prototype_2d=None,
        sample_class_ids=[0, 1] * 10,
        prototype_class_ids=None,
        class_names=["a", "b"],
        suptitle="stage 3",
        save_path=path,
    )

    assert path.exists()


def test_the_comparison_still_plots_with_prototypes(tmp_path):
    # Stage 2's call must keep working unchanged.
    path = tmp_path / "with_prototypes.png"

    plot_feature_space_comparison(
        _panels(),
        np.zeros((2, 2)),
        [0, 1] * 10,
        [0, 1],
        ["a", "b"],
        "stage 2",
        path,
    )

    assert path.exists()


def test_giving_only_one_prototype_argument_raises(tmp_path):
    with pytest.raises(ValueError, match="both be None"):
        plot_feature_space_comparison(
            _panels(),
            prototype_2d=np.zeros((2, 2)),
            sample_class_ids=[0, 1] * 10,
            prototype_class_ids=None,
            class_names=["a", "b"],
            suptitle="broken",
            save_path=tmp_path / "f.png",
        )


# --- run loading ---


def test_a_missing_run_returns_none(tmp_path):
    assert load_stage3_run(tmp_path, "dtd", "dinov2_vits14", "fm_cls_rolled", 0) is None


def test_missing_runs_skip_the_figures_rather_than_failing(tmp_path):
    assert plot_stage3_training_curves(
        "dtd", "dinov2_vits14", tmp_path, tmp_path
    ) is None


def test_the_run_directory_uses_this_stages_fixed_k_and_t(tmp_path):
    run_dir = stage3_run_directory(tmp_path, "dtd", "dinov2_vits14", "fm_cls_guided", 2)

    assert f"k{STAGE3_K_SHOT}" in run_dir.parts
    assert "T4" in run_dir.parts
    assert run_dir.name == "seed2"


def test_every_method_has_a_readable_panel_title():
    assert set(METHOD_PANEL_TITLES) == set(STAGE3_METHODS)


# --- Integration against the real completed runs ---


def _skip_unless_complete(dataset, encoder):
    if not (REAL_CACHE / dataset / encoder / "test.pt").exists():
        pytest.skip(f"no cached features for {dataset}/{encoder}")
    for method in STAGE3_METHODS:
        if load_stage3_run(REAL_OUTPUTS, dataset, encoder, method, 0) is None:
            pytest.skip(f"Stage 3 run not completed for {dataset}/{encoder}/{method}")


@pytest.mark.parametrize("dataset, encoder", list(STAGE3_SETTINGS.items()))
def test_real_runs_load_with_their_own_architecture(dataset, encoder):
    _skip_unless_complete(dataset, encoder)

    for method in STAGE3_METHODS:
        run = load_stage3_run(REAL_OUTPUTS, dataset, encoder, method, 0)
        assert run["hidden_dims"] == [512, 512]
        assert run["num_euler_steps"] == 4
        assert len(run["history"]) == 200
        assert "initial_val_accuracy" in run["result"]


@pytest.mark.parametrize("dataset, encoder", list(STAGE3_SETTINGS.items()))
def test_the_feature_space_figure_reuses_the_stage_1_selection(dataset, encoder):
    # part_3.pdf: "the same test examples and class colors across all
    # comparisons". Reusing the saved selection also keeps the Stage 3 figure
    # comparable to the Stage 1 and Stage 2 ones.
    _skip_unless_complete(dataset, encoder)
    selection_path = REAL_REPORTS / f"feature_viz_selection_{dataset}.json"
    if not selection_path.exists():
        pytest.skip("no saved selection")

    before = selection_path.read_text()
    plot_stage3_feature_space(
        dataset, encoder, REAL_CACHE, PROJECT_ROOT / "data", REAL_OUTPUTS,
        REAL_REPORTS, PROJECT_ROOT / "reports" / "figures", CPU,
    )

    assert selection_path.read_text() == before  # reused, not regenerated


# --- class separation ---


def _two_clusters(separation: float, spread: float = 1.0, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(50, 4, generator=generator) * spread
    b = torch.randn(50, 4, generator=generator) * spread + separation
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    return torch.cat([a, b]), labels


def test_separation_rises_as_classes_move_apart():
    close, labels = _two_clusters(separation=1.0)
    far, _ = _two_clusters(separation=10.0)

    assert class_separation(far, labels) > class_separation(close, labels)


def test_separation_falls_as_classes_spread_out():
    tight, labels = _two_clusters(separation=5.0, spread=0.5)
    loose, _ = _two_clusters(separation=5.0, spread=3.0)

    assert class_separation(tight, labels) > class_separation(loose, labels)


def test_separation_is_scale_invariant():
    # Between- and within-class scatter both scale with the square of the
    # feature magnitude, so the ratio does not - which is what makes it
    # comparable between the original features and the transported ones.
    features, labels = _two_clusters(separation=3.0)

    assert class_separation(features * 7.0, labels) == pytest.approx(
        class_separation(features, labels), rel=1e-4
    )


def test_separation_is_translation_invariant():
    features, labels = _two_clusters(separation=3.0)
    shifted = features + 100.0

    assert class_separation(shifted, labels) == pytest.approx(
        class_separation(features, labels), rel=1e-4
    )


# --- the separation table ---


def _separation_rows():
    return [
        {"dataset": "dtd", "encoder": "e", "method": "original",
         "separation": 0.4439, "displacement": 0.0, "feature_norm": 47.8},
        {"dataset": "dtd", "encoder": "e", "method": "fm_cls_rolled",
         "separation": 0.4528, "displacement": 2.16, "feature_norm": 47.8},
        {"dataset": "dtd", "encoder": "e", "method": "fm_cls_guided",
         "separation": 0.4810, "displacement": 6.20, "feature_norm": 47.8},
    ]


def test_the_separation_table_reports_change_against_the_original():
    lines = format_class_separation_table(_separation_rows()).strip().splitlines()

    assert lines[2].endswith("| 0.00 | 0.0% |")  # original has no change
    assert "+2.0%" in lines[3]
    assert "+8.4%" in lines[4]
    assert "13.0%" in lines[4]  # displacement as a fraction of the feature norm


def test_an_empty_separation_table_says_so():
    assert "No Stage 3 runs" in format_class_separation_table([])


# --- outcome counting and the observations ---


def _accuracy_summary(dataset, method, delta):
    return {
        "dataset": dataset, "encoder": "e", "method": method, "k_shot": STAGE3_K_SHOT,
        "num_runs": 3, "mean_test_accuracy": 0.7 + delta,
        "std_test_accuracy": 0.005, "mean_delta_accuracy": delta,
        "std_delta_accuracy": 0.002,
    }


SETTINGS = [("dtd", "e"), ("flowers102", "e")]


def test_outcomes_count_improvements_and_wins():
    summaries = [
        _accuracy_summary("dtd", "fm_cls_rolled", 0.002),
        _accuracy_summary("dtd", "fm_cls_guided", 0.010),
        _accuracy_summary("flowers102", "fm_cls_rolled", 0.009),
        _accuracy_summary("flowers102", "fm_cls_guided", -0.001),
    ]

    counts = summarize_stage3_outcomes(summaries, SETTINGS)

    assert counts["improved"] == {"fm_cls_rolled": 2, "fm_cls_guided": 1}
    assert counts["total"] == {"fm_cls_rolled": 2, "fm_cls_guided": 2}
    assert counts["wins"] == {"fm_cls_rolled": 1, "fm_cls_guided": 1}


def test_the_ranking_sentence_says_split_when_the_methods_split():
    summaries = [
        _accuracy_summary("dtd", "fm_cls_rolled", 0.002),
        _accuracy_summary("dtd", "fm_cls_guided", 0.010),
        _accuracy_summary("flowers102", "fm_cls_rolled", 0.009),
        _accuracy_summary("flowers102", "fm_cls_guided", 0.001),
    ]

    text = " ".join(
        format_stage3_observations(summaries, [], _separation_rows(), SETTINGS)
    )

    assert "differs by dataset" in text
    assert "neither strategy dominates" in text


def test_the_ranking_sentence_says_so_when_one_method_wins_everywhere():
    # Regression test. An earlier version asserted "the stronger method
    # differs by dataset" unconditionally, which contradicted the table
    # beside it when one method won both settings.
    summaries = [
        _accuracy_summary("dtd", "fm_cls_rolled", 0.002),
        _accuracy_summary("dtd", "fm_cls_guided", 0.010),
        _accuracy_summary("flowers102", "fm_cls_rolled", 0.009),
        _accuracy_summary("flowers102", "fm_cls_guided", 0.010),
    ]

    text = " ".join(
        format_stage3_observations(summaries, [], _separation_rows(), SETTINGS)
    )

    assert "fm_cls_guided` is the stronger method on every setting" in text
    assert "differs by dataset" not in text


def test_the_observations_quote_the_measured_separation_changes():
    summaries = [_accuracy_summary("dtd", "fm_cls_guided", 0.010)]

    text = " ".join(
        format_stage3_observations(summaries, [], _separation_rows(), [("dtd", "e")])
    )

    assert "+8.4%" in text  # derived from the rows, not hardcoded
    assert "13.0%" in text


# --- the assembled section ---


def test_the_section_contains_every_required_part(tmp_path):
    baseline = _accuracy_summary("dtd", "linear_probe", 0.0)
    baseline["mean_delta_accuracy"] = None  # the baseline has nothing to differ from
    baseline["std_delta_accuracy"] = None
    summaries = [
        baseline,
        _accuracy_summary("dtd", "fm_cls_rolled", 0.002),
        _accuracy_summary("dtd", "fm_cls_guided", 0.010),
    ]

    text = "\n".join(
        format_stage3_section(
            summaries, [], _separation_rows(), [("dtd", "e")],
            Stage3Figures(), tmp_path,
        )
    )

    assert "# Stage 3 Results: FM Before a Linear Classifier" in text
    assert "## Main comparison" in text
    assert "## Training and selection diagnostics" in text
    assert "## Class structure in the full feature space" in text
    assert "## Observations" in text
    assert "### Caveats" in text


def test_the_section_embeds_the_figures_relative_to_the_project_root(tmp_path):
    figures = Stage3Figures()
    figure_path = tmp_path / "reports" / "figures" / "curves.png"
    figure_path.parent.mkdir(parents=True)
    figure_path.write_bytes(b"")
    figures.curves.append(("dtd", "dinov2_vits14", figure_path))

    text = "\n".join(
        format_stage3_section(
            [], [], [], [("dtd", "dinov2_vits14")], figures, tmp_path
        )
    )

    assert "![dtd dinov2_vits14](reports/figures/curves.png)" in text


def test_empty_figure_sections_are_omitted(tmp_path):
    text = "\n".join(
        format_stage3_section([], [], [], [], Stage3Figures(), tmp_path)
    )

    assert "Stage 3: training curves" not in text
    assert "Stage 3: feature space" not in text


def test_the_figure_sections_are_ordered_curves_then_feature_space():
    headings = [heading for heading, _ in Stage3Figures().sections()]

    assert "training curves" in headings[0]
    assert "feature space" in headings[1]


def test_a_missing_tuning_file_contributes_nothing(tmp_path):
    assert _tuning_tables(tmp_path / "absent.json") == []


def test_the_tuning_section_renders_when_the_file_exists():
    tuning_path = REAL_REPORTS / "stage3_tuning.json"
    if not tuning_path.exists():
        pytest.skip("no tuning results")

    lines = _tuning_tables(tuning_path, top_n=3)
    text = "\n".join(lines)

    assert "## Hyperparameter search" in text
    assert "never used for ranking" in text
    assert "### fm_cls_guided on dtd" in text

