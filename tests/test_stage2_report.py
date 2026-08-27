from pathlib import Path

import pytest

from src.evaluation.stage2_report import (
    Stage2Figures,
    format_observations,
    format_stage2_section,
    summarize_outcomes,
)

PAIRS = [("dtd", "resnet18")]


def _summary(method, num_euler_steps, k_shot, accuracy, delta, runs=3):
    return {
        "dataset": "dtd", "encoder": "resnet18", "method": method, "k_shot": k_shot,
        "num_euler_steps": num_euler_steps, "num_runs": runs,
        "mean_test_accuracy": accuracy,
        "std_test_accuracy": 0.01 if runs > 1 else None,
        "mean_baseline_accuracy": None if delta is None else accuracy - delta,
        "mean_delta_accuracy": delta,
        "std_delta_accuracy": 0.005 if (delta is not None and runs > 1) else None,
        "seed_accuracies": {},
    }


def _full_grid(standard_delta, rolled_delta):
    """Baseline plus both FM methods at both T, across all three K."""
    summaries = [_summary("prototype", None, k, 0.50, None) for k in (5, 10, "full")]
    for k_shot in (5, 10, "full"):
        for num_euler_steps in (4, 12):
            summaries.append(
                _summary("fm_standard", num_euler_steps, k_shot, 0.50 + standard_delta, standard_delta)
            )
            summaries.append(
                _summary("fm_rolled", num_euler_steps, k_shot, 0.50 + rolled_delta, rolled_delta)
            )
    return summaries


# --- counted outcomes ---


def test_counts_every_matched_comparison():
    # 3 K values x 2 step counts = 6 matched standard-vs-rolled comparisons.
    summaries = _full_grid(standard_delta=0.02, rolled_delta=-0.05)

    counts = summarize_outcomes(summaries, PAIRS)

    assert counts["comparisons"] == 6
    assert counts["standard_beats_rolled"] == 6


def test_counts_reflect_which_variant_actually_won():
    summaries = _full_grid(standard_delta=-0.05, rolled_delta=0.02)

    counts = summarize_outcomes(summaries, PAIRS)

    assert counts["standard_beats_rolled"] == 0


def test_improvements_are_counted_against_the_baseline():
    summaries = _full_grid(standard_delta=0.02, rolled_delta=-0.05)

    counts = summarize_outcomes(summaries, PAIRS)

    assert counts["improved"]["fm_standard"] == 6
    assert counts["improved"]["fm_rolled"] == 0
    assert counts["total"]["fm_standard"] == 6


def test_full_data_settings_are_counted_separately():
    summaries = [_summary("prototype", None, k, 0.50, None) for k in (5, 10, "full")]
    for num_euler_steps in (4, 12):
        # Negative in few-shot, positive at full - the real pattern.
        summaries.append(_summary("fm_standard", num_euler_steps, 5, 0.48, -0.02))
        summaries.append(_summary("fm_standard", num_euler_steps, 10, 0.49, -0.01))
        summaries.append(_summary("fm_standard", num_euler_steps, "full", 0.52, 0.02, runs=1))

    counts = summarize_outcomes(summaries, PAIRS)

    assert counts["improved"]["fm_standard"] == 2
    assert counts["total"]["fm_standard"] == 6
    assert counts["improved_full"]["fm_standard"] == 2
    assert counts["total_full"]["fm_standard"] == 2


def test_stage_1_settings_are_ignored_by_the_counts():
    # The prototype and linear-probe rows carry no delta and must not be
    # counted as flow-matching outcomes.
    summaries = [
        _summary("prototype", None, 5, 0.50, None),
        _summary("linear_probe", None, 5, 0.45, None),
    ]

    counts = summarize_outcomes(summaries, PAIRS)

    assert counts["total"]["fm_standard"] == 0
    assert counts["comparisons"] == 0


# --- the written observations ---


def test_observations_state_the_measured_counts():
    summaries = _full_grid(standard_delta=0.02, rolled_delta=-0.05)

    text = "\n".join(format_observations(summaries, PAIRS))

    assert "6 of 6 matched comparisons" in text
    assert "## Observations" in text


def test_observations_track_the_data_rather_than_being_hardcoded():
    # The same prose rendered against a reversed outcome must report the
    # reversed counts.
    won = "\n".join(format_observations(_full_grid(0.02, -0.05), PAIRS))
    lost = "\n".join(format_observations(_full_grid(-0.05, 0.02), PAIRS))

    assert "6 of 6 matched comparisons" in won
    assert "0 of 6 matched comparisons" in lost


def test_observations_include_the_run_count_caveat():
    summaries = _full_grid(standard_delta=0.02, rolled_delta=-0.05)

    text = "\n".join(format_observations(summaries, PAIRS))

    assert "Caveats" in text
    assert "single runs" in text
    assert "Flowers-102" in text


# --- the assembled section ---


def test_section_contains_tables_observations_and_figures(tmp_path):
    summaries = _full_grid(standard_delta=0.02, rolled_delta=-0.05)
    figure_path = tmp_path / "reports" / "figures" / "stage2_accuracy_vs_shot_dtd_resnet18.png"
    figure_path.parent.mkdir(parents=True)
    figure_path.write_bytes(b"")
    figures = Stage2Figures(accuracy_vs_shot=[("dtd", "resnet18", figure_path)])

    text = "\n".join(format_stage2_section(summaries, PAIRS, figures, tmp_path))

    assert "# Stage 2 Results" in text
    assert "## Comparison tables" in text
    assert "## Observations" in text
    assert "Stage 2: accuracy vs. training-set size" in text
    assert "reports/figures/stage2_accuracy_vs_shot_dtd_resnet18.png" in text


def test_section_uses_forward_slashes_in_image_paths(tmp_path):
    # RESULTS.md must render on any platform, including when generated on
    # Windows where Path renders backslashes.
    summaries = _full_grid(standard_delta=0.02, rolled_delta=-0.05)
    figure_path = tmp_path / "reports" / "figures" / "fig.png"
    figure_path.parent.mkdir(parents=True)
    figure_path.write_bytes(b"")
    figures = Stage2Figures(per_step=[("dtd (K=10)", "resnet18", figure_path)])

    text = "\n".join(format_stage2_section(summaries, PAIRS, figures, tmp_path))

    assert "](reports/figures/fig.png)" in text
    assert "\\" not in text.split("](")[1].split(")")[0]


def test_empty_figure_sections_are_omitted(tmp_path):
    summaries = _full_grid(standard_delta=0.02, rolled_delta=-0.05)

    text = "\n".join(format_stage2_section(summaries, PAIRS, Stage2Figures(), tmp_path))

    assert "## Comparison tables" in text
    assert "Stage 2: flow trajectories" not in text


def test_figure_sections_are_listed_in_a_stable_order():
    figures = Stage2Figures()
    headings = [heading for heading, _ in figures.sections()]

    assert headings[0].endswith("accuracy vs. training-set size")
    assert "flow trajectories" in headings[4]
    assert "reverse flow" in headings[5]
    assert "metrics along the flow" in headings[6]


# --- integration against the real report, when it has been generated ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_REAL_REPORT = PROJECT_ROOT / "RESULTS.md"


@pytest.mark.skipif(not _REAL_REPORT.exists(), reason="RESULTS.md has not been generated")
def test_generated_report_contains_both_stages():
    text = _REAL_REPORT.read_text(encoding="utf-8")

    assert "# Stage 1 Results" in text
    assert "# Stage 2 Results" in text
    assert "## Observations" in text


@pytest.mark.skipif(not _REAL_REPORT.exists(), reason="RESULTS.md has not been generated")
def test_every_image_referenced_by_the_report_exists():
    import re

    text = _REAL_REPORT.read_text(encoding="utf-8")
    referenced = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)

    assert referenced, "report references no images"
    missing = [path for path in referenced if not (PROJECT_ROOT / path).exists()]
    assert not missing, f"report references missing figures: {missing}"
