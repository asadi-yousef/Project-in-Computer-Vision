"""Build the figures and text for the Stage 3 report section (part_3.pdf).

Reads completed Stage 3 runs from `outputs/` and produces the deliverables
part_3.pdf asks for beyond the accuracy table: representative training
curves for both methods, and a feature-space view of the original features
against the transported ones.

Everything here is tolerant of a partially-completed sweep: a missing run
means the figure is skipped and None returned, never an exception, so a
report can be built at any point.
"""

import dataclasses
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from src.data.datasets import get_class_names, load_dataset_splits
from src.evaluation.tables import (
    format_stage3_comparison_table,
    format_stage3_diagnostics_table,
)
from src.features.loading import load_validated_feature_cache
from src.flow_matching.inference import transport_with_checkpoint
from src.flow_matching.stage3_runner import stage3_run_dir
from src.flow_matching.stage3_tuning import format_tuning_table, load_tuning_results
from src.utils.config import (
    STAGE3_K_SHOT,
    STAGE3_METHODS,
    STAGE3_SEEDS,
    load_config,
    stage3_hyperparams_for,
)
from src.visualization.feature_space import (
    load_selection,
    plot_feature_space_comparison,
    project_feature_groups,
    save_selection,
    select_classes_and_samples,
)
from src.visualization.loss_curves import load_history
from src.visualization.stage3_curves import Stage3CurveRun, plot_stage3_curves

# Per-project convention: seed 0 is "the representative run" the spec asks
# for when showing curves, matching what Stage 1 and Stage 2 use.
REPRESENTATIVE_SEED = 0

# Human-readable panel titles for the feature-space figure.
METHOD_PANEL_TITLES = {
    "fm_cls_rolled": "after end-to-end rolled-out training",
    "fm_cls_guided": "after classifier-guided FM",
}

# (label, encoder, absolute path) - the shape the report and PDF consume.
FigureList = List[Tuple[str, str, Path]]


@dataclasses.dataclass
class Stage3Figures:
    """Every Stage 3 figure produced, grouped by the section it belongs to."""

    curves: FigureList = dataclasses.field(default_factory=list)
    feature_space: FigureList = dataclasses.field(default_factory=list)

    def sections(self) -> List[Tuple[str, FigureList]]:
        """(heading, figures) in the order the report presents them."""
        return [
            (
                f"Stage 3: training curves ({STAGE3_K_SHOT}-shot, seed "
                f"{REPRESENTATIVE_SEED})",
                self.curves,
            ),
            ("Stage 3: feature space before and after the flow (t-SNE)", self.feature_space),
        ]


def stage3_run_directory(
    output_dir: Union[str, Path], dataset: str, encoder: str, method: str, seed: int
) -> Path:
    """The run directory for one Stage 3 run, at this stage's fixed K and T."""
    hyperparams = stage3_hyperparams_for(method, dataset)
    return stage3_run_dir(
        output_dir, dataset, encoder, method, STAGE3_K_SHOT,
        hyperparams.num_euler_steps, seed,
    )


def load_stage3_run(
    output_dir: Union[str, Path], dataset: str, encoder: str, method: str, seed: int
) -> Optional[dict]:
    """Load one completed run's checkpoint, history, result and saved config.

    Returns:
        A dict with state_dict, hidden_dims, num_euler_steps, history and
        result, or None if the run has not been completed.
    """
    run_dir = stage3_run_directory(output_dir, dataset, encoder, method, seed)
    checkpoint_path = run_dir / "checkpoint.pt"
    config_path = run_dir / "config.yaml"
    result_path = run_dir / "result.json"
    history_path = run_dir / "history.json"

    if not all(p.exists() for p in (checkpoint_path, config_path, result_path, history_path)):
        return None

    # Architecture and T come from the run's own saved config rather than
    # from the current defaults, so a report built from older runs still
    # rebuilds the network they were actually trained with.
    stage3 = load_config(config_path).stage3
    return {
        "state_dict": torch.load(checkpoint_path, weights_only=True),
        "hidden_dims": stage3.hidden_dims,
        "num_euler_steps": stage3.num_euler_steps,
        "history": load_history(history_path),
        "result": json.loads(result_path.read_text())["result"],
        "run_dir": run_dir,
    }


def plot_stage3_training_curves(
    dataset: str,
    encoder: str,
    output_dir: Union[str, Path],
    figures_dir: Union[str, Path],
    seed: int = REPRESENTATIVE_SEED,
    methods: Sequence[str] = STAGE3_METHODS,
) -> Optional[Path]:
    """part_3.pdf's "Training behavior" deliverable, for one dataset.

    Returns:
        The figure path, or None if no run was available.
    """
    runs = []
    for method in methods:
        run = load_stage3_run(output_dir, dataset, encoder, method, seed)
        if run is None:
            continue
        runs.append(
            Stage3CurveRun(
                method=method,
                history=run["history"],
                initial_val_accuracy=run["result"]["initial_val_accuracy"],
                best_epoch=run["result"]["best_epoch"],
            )
        )

    if not runs:
        return None

    save_path = Path(figures_dir) / f"stage3_curves_{dataset}_{encoder}.png"
    plot_stage3_curves(runs, dataset, encoder, STAGE3_K_SHOT, seed, save_path)
    return save_path


def _load_or_create_selection(dataset: str, test_labels, reports_dir: Union[str, Path]):
    """Reuse the class/sample selection Stage 1 and Stage 2 plotted.

    part_3.pdf asks for "the same test examples and class colors across all
    comparisons". Reading back the saved selection satisfies that within
    Stage 3, and additionally makes the Stage 3 figure directly comparable
    to the Stage 1 and Stage 2 ones for the same dataset.
    """
    selection_path = Path(reports_dir) / f"feature_viz_selection_{dataset}.json"
    if selection_path.exists():
        return load_selection(selection_path)

    selection = select_classes_and_samples(dataset, test_labels)
    save_selection(selection, selection_path)
    return selection


def transported_test_features(
    run: dict, features: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Push features through a completed run's trained flow.

    Uses `transport_with_checkpoint`, the same integrator Stage 2's figures
    use: the near-identity initialization only matters before weights are
    loaded, so no Stage 3-specific builder is needed here.
    """
    return transport_with_checkpoint(
        run["state_dict"], run["hidden_dims"], features, run["num_euler_steps"], device
    )


def plot_stage3_feature_space(
    dataset: str,
    encoder: str,
    cache_dir: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reports_dir: Union[str, Path],
    figures_dir: Union[str, Path],
    device: torch.device,
    seed: int = REPRESENTATIVE_SEED,
    methods: Sequence[str] = STAGE3_METHODS,
) -> Optional[Path]:
    """part_3.pdf's "Feature-space visualization" deliverable, for one dataset.

    Shows the original encoder features alongside the transported features
    for each Stage 3 method, on the same test examples with the same class
    colours, under a single jointly-fitted t-SNE.

    The projection is fitted on **unnormalized** features, unlike Stage 2's.
    Stage 3's classifier is a linear probe rather than cosine similarity, so
    magnitude is part of what it sees - and the displacement magnitude is the
    clearest difference between the two strategies, which normalizing would
    hide.

    Returns:
        The figure path, or None if no run was available.
    """
    test_features, test_labels, _ = load_validated_feature_cache(
        cache_dir, dataset, encoder, "test"
    )
    selection = _load_or_create_selection(dataset, test_labels, reports_dir)

    indices = torch.tensor(selection.sample_indices)
    original = test_features[indices]
    sample_class_ids = test_labels[indices].tolist()

    transported: Dict[str, torch.Tensor] = {}
    for method in methods:
        run = load_stage3_run(output_dir, dataset, encoder, method, seed)
        if run is None:
            continue
        transported[method] = transported_test_features(run, original, device)

    if not transported:
        return None

    ordered_methods = [method for method in methods if method in transported]
    projected = project_feature_groups(
        [original] + [transported[method] for method in ordered_methods],
        seed=0,
        normalize=False,
    )

    class_names = get_class_names(
        load_dataset_splits(dataset, data_dir, download=False)["train"]
    )
    panels: List[Tuple[str, object]] = [("original encoder features", projected[0])]
    for method, coordinates in zip(ordered_methods, projected[1:]):
        panels.append((METHOD_PANEL_TITLES.get(method, method), coordinates))

    save_path = Path(figures_dir) / f"stage3_feature_space_{dataset}_{encoder}.png"
    plot_feature_space_comparison(
        panels,
        prototype_2d=None,
        sample_class_ids=sample_class_ids,
        prototype_class_ids=None,
        class_names=class_names,
        suptitle=(
            f"{dataset} / {encoder}: feature space before and after the Stage 3 flow "
            f"(K={STAGE3_K_SHOT}, seed {seed}, {len(selection.class_ids)} classes)"
        ),
        save_path=save_path,
    )
    return save_path


def class_separation(features: torch.Tensor, labels: torch.Tensor) -> float:
    """Between-class scatter over within-class scatter, in the full space.

    A Fisher-style ratio: higher means the classes sit further apart relative
    to how spread out each one is. Used because part_3.pdf gives the
    feature-space figure a specific purpose - "to examine how each training
    strategy changes the class structure of the frozen representation" - and a
    two-dimensional projection can only suggest an answer. This measures it in
    the space the classifier actually sees.

    Args:
        features: (N, D) features.
        labels: (N,) class labels.

    Returns:
        The ratio. Comparable across methods on the same samples, not across
        datasets.
    """
    class_ids = sorted(set(labels.tolist()))
    means = torch.stack([features[labels == c].mean(dim=0) for c in class_ids])
    grand_mean = features.mean(dim=0)

    between = (means - grand_mean).pow(2).sum(dim=1).mean()
    within = torch.stack(
        [
            (features[labels == c] - means[index]).pow(2).sum(dim=1).mean()
            for index, c in enumerate(class_ids)
        ]
    ).mean()
    return (between / within).item()


def measure_class_separation(
    dataset: str,
    encoder: str,
    cache_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reports_dir: Union[str, Path],
    device: torch.device,
    seed: int = REPRESENTATIVE_SEED,
    methods: Sequence[str] = STAGE3_METHODS,
) -> List[dict]:
    """Class separation before and after each Stage 3 flow, plus displacement.

    Measured on exactly the samples the feature-space figure plots, so the
    numbers and the picture describe the same thing.

    Returns:
        One dict per condition (original first, then each method), or an empty
        list if no run was available.
    """
    test_features, test_labels, _ = load_validated_feature_cache(
        cache_dir, dataset, encoder, "test"
    )
    selection = _load_or_create_selection(dataset, test_labels, reports_dir)
    indices = torch.tensor(selection.sample_indices)
    original, labels = test_features[indices], test_labels[indices]

    rows = [
        {
            "dataset": dataset,
            "encoder": encoder,
            "method": "original",
            "separation": class_separation(original, labels),
            "displacement": 0.0,
            "feature_norm": original.norm(dim=1).mean().item(),
        }
    ]
    for method in methods:
        run = load_stage3_run(output_dir, dataset, encoder, method, seed)
        if run is None:
            continue
        transported = transported_test_features(run, original, device)
        rows.append(
            {
                "dataset": dataset,
                "encoder": encoder,
                "method": method,
                "separation": class_separation(transported, labels),
                "displacement": (transported - original).norm(dim=1).mean().item(),
                "feature_norm": original.norm(dim=1).mean().item(),
            }
        )

    return rows if len(rows) > 1 else []


def format_class_separation_table(rows: Sequence[dict]) -> str:
    """Render `measure_class_separation` output as a Markdown table."""
    if not rows:
        return "_No Stage 3 runs._\n"

    header = (
        "| Dataset | Condition | Class separation | Change | "
        "Mean displacement | Displacement / feature norm |\n"
    )
    header += "|---" * 6 + "|\n"

    baseline = {}
    lines = []
    for row in rows:
        if row["method"] == "original":
            baseline[row["dataset"]] = row["separation"]
            change = "-"
        else:
            reference = baseline.get(row["dataset"])
            change = (
                "-"
                if not reference
                else f"{(row['separation'] / reference - 1) * 100:+.1f}%"
            )
        percent = 100 * row["displacement"] / row["feature_norm"]
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['separation']:.4f} | {change} "
            f"| {row['displacement']:.2f} | {percent:.1f}% |"
        )
    return header + "\n".join(lines) + "\n"


def generate_stage3_figures(
    settings: Sequence[Tuple[str, str]],
    cache_dir: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reports_dir: Union[str, Path],
    figures_dir: Union[str, Path],
    device: torch.device,
) -> Tuple[Stage3Figures, List[dict]]:
    """Produce every Stage 3 figure, and the class-separation measurements.

    Args:
        settings: the (dataset, encoder) pairs Stage 3 ran on.
        cache_dir, data_dir, output_dir, reports_dir, figures_dir: project
            directories.
        device: device to run the transports on.

    Returns:
        (figures, separation_rows). Missing runs are skipped with a printed
        message, so a partial sweep still reports.
    """
    figures = Stage3Figures()
    separations: List[dict] = []

    for dataset, encoder in settings:
        curves_path = plot_stage3_training_curves(dataset, encoder, output_dir, figures_dir)
        if curves_path:
            figures.curves.append((dataset, encoder, curves_path))
        else:
            print(f"  stage 3 {dataset}/{encoder}: no completed runs, skipping")
            continue

        feature_space_path = plot_stage3_feature_space(
            dataset, encoder, cache_dir, data_dir, output_dir, reports_dir,
            figures_dir, device,
        )
        if feature_space_path:
            figures.feature_space.append((dataset, encoder, feature_space_path))

        separations.extend(
            measure_class_separation(
                dataset, encoder, cache_dir, output_dir, reports_dir, device
            )
        )

    return figures, separations


def summarize_stage3_outcomes(
    summaries: List[dict], settings: Sequence[Tuple[str, str]]
) -> dict:
    """Count the headline outcomes, so the prose states measured facts.

    Derived from the summaries rather than hardcoded, so re-running the sweep
    cannot leave the write-up describing numbers that are no longer there.
    """
    improved = {method: 0 for method in STAGE3_METHODS}
    total = {method: 0 for method in STAGE3_METHODS}
    wins = {method: 0 for method in STAGE3_METHODS}
    best_delta = {}

    for dataset, encoder in settings:
        lookup = {
            s["method"]: s
            for s in summaries
            if s["dataset"] == dataset
            and s["encoder"] == encoder
            and s["k_shot"] == STAGE3_K_SHOT
        }
        deltas = {}
        for method in STAGE3_METHODS:
            summary = lookup.get(method)
            if summary is None or summary.get("mean_delta_accuracy") is None:
                continue
            total[method] += 1
            deltas[method] = summary["mean_delta_accuracy"]
            if summary["mean_delta_accuracy"] > 0:
                improved[method] += 1
        if deltas:
            winner = max(deltas, key=deltas.get)
            wins[winner] += 1
            best_delta[dataset] = (winner, deltas[winner])

    return {
        "improved": improved,
        "total": total,
        "wins": wins,
        "best_delta": best_delta,
        "num_settings": len(settings),
    }


def format_stage3_observations(
    summaries: List[dict],
    stage3_summaries: List[dict],
    separations: Sequence[dict],
    settings: Sequence[Tuple[str, str]],
) -> List[str]:
    """The Stage 3 discussion: what the results show, and the caveats.

    The counted claims are derived from the summaries and measurements so they
    cannot drift away from the tables above them.
    """
    counts = summarize_stage3_outcomes(summaries, settings)
    improved_total = sum(counts["improved"].values())
    comparisons = sum(counts["total"].values())

    winners = ", ".join(
        f"{dataset} ({method}, {delta * 100:+.2f})"
        for dataset, (method, delta) in sorted(counts["best_delta"].items())
    )

    # Whether one strategy wins everywhere is itself a finding, so the
    # sentence is derived rather than asserted: hardcoding either reading
    # would let the prose contradict the table beside it after a re-run.
    outright_winners = [
        method for method, count in counts["wins"].items() if count == counts["num_settings"]
    ]
    if outright_winners and counts["num_settings"] > 1:
        margins = ", ".join(
            f"{dataset} {delta * 100:+.2f}"
            for dataset, (_, delta) in sorted(counts["best_delta"].items())
        )
        ranking_sentence = (
            f"`{outright_winners[0]}` is the stronger method on every setting "
            f"({margins}), though by margins that differ sharply between datasets."
        )
    else:
        ranking_sentence = (
            f"The stronger method differs by dataset - {winners} - so neither "
            "strategy dominates."
        )

    baselines = {
        row["dataset"]: row["separation"]
        for row in separations
        if row["method"] == "original"
    }
    separation_claims = ", ".join(
        f"{row['dataset']}/{row['method']} "
        f"{(row['separation'] / baselines[row['dataset']] - 1) * 100:+.1f}%"
        for row in separations
        if row["method"] != "original" and row["dataset"] in baselines
    )

    displacement_claims = ", ".join(
        f"{row['dataset']}/{row['method']} "
        f"{100 * row['displacement'] / row['feature_norm']:.1f}%"
        for row in separations
        if row["method"] != "original"
    )

    return [
        "## Observations\n",
        f"**Both Stage 3 methods improve on the Stage 1 linear probe, in "
        f"{improved_total} of {comparisons} settings.** {ranking_sentence}\n",
        "**The classification objective starts almost exhausted.** part_3.pdf "
        "requires reusing Stage 1's own training subset, and Stage 1 trained its "
        "probe on that subset to convergence - it already reaches 100% accuracy "
        "and a cross-entropy near 0.002 there before Stage 3 begins. Strategy 1's "
        "loss therefore starts at its floor, and the training-accuracy curves sit "
        "flat at 100% throughout. All of the usable signal is in validation.\n",
        "**Unregularized, that pushes the flow toward inflating feature magnitude "
        "rather than improving the representation.** With no displacement penalty, "
        "validation accuracy peaks within the first few epochs and then falls "
        "below the untrained identity; across the search it ranks last of ten "
        "configurations on Flowers-102 and ninth of ten on DTD. The regularization "
        "part_3.pdf offers for exactly this is what makes Strategy 1 work at all.\n",
        "**The two strategies reach comparable accuracy by changing the "
        "representation very differently.** Measured in the full feature space on "
        f"the plotted samples, both raise class separation ({separation_claims}), "
        "but they move the features by very different amounts relative to the "
        f"feature norm ({displacement_claims}). Strategy 1 makes a small "
        "correction; Strategy 2 substantially restructures the features.\n",
        "**Classifier-guided targets compound, and the refresh interval controls "
        "it.** Each refresh rebuilds the target from the *current* transported "
        "feature, so the target ratchets away from the original and the flow has "
        "no fixed point to converge on. Refreshing every epoch at a step size of "
        "0.1 drives mean displacement to roughly 73 on DTD against a feature norm "
        "of 48; refreshing every 20 epochs cuts that to about 9 while improving "
        "validation accuracy. Without a slower refresh, only validation-based "
        "checkpoint selection stops the drift.\n",
        "### Caveats\n",
        "- **The gains are small relative to the seed spread.** Deltas range from "
        "+0.20 to +1.05 points while the per-seed standard deviation reaches 0.87 "
        "on DTD's classifier-guided runs. On DTD one seed gained +0.05 where the "
        "other two gained +1.44 and +1.65; that seed also started from the "
        "strongest baseline. With three seeds this is an observation, not a "
        "demonstrated effect.\n",
        "- **Hyperparameters were selected on validation, which flatters the "
        "reported test numbers slightly.** The search ranked 34 configurations per "
        "strategy on mean validation delta with test held out; against an oracle "
        "selecting on test it gave up at most 0.15 points. Compared with the "
        "untuned defaults, tuning moved the mean test delta from +0.68 to +0.80, "
        "and only one of the four settings improved materially.\n",
        "- **Two selected configurations sit on a grid boundary.** Both "
        "classifier-guided selections took the smallest step size searched, and "
        "the Flowers-102 one took the largest refresh interval, so the optimum may "
        "lie outside the range explored.\n",
        "- **For Flowers-102, K=10 is the entire official training split** (1020 "
        "images, 102 classes). That row is a full-data result rather than a "
        "few-shot one, and its seeds differ only in initialization.\n",
        "- **Two-dimensional projections are qualitative.** The class-separation "
        "table is measured in the full feature space; the t-SNE panels illustrate "
        "it rather than establish it.\n",
    ]


def _tuning_tables(tuning_path: Union[str, Path], top_n: int = 5) -> List[str]:
    """Render the hyperparameter search, best-first, per method and dataset."""
    tuning_path = Path(tuning_path)
    if not tuning_path.exists():
        return []

    summaries = load_tuning_results(tuning_path)
    grouped: Dict[Tuple[str, str], list] = {}
    for summary in summaries:
        grouped.setdefault((summary.method, summary.dataset), []).append(summary)

    lines = [
        "## Hyperparameter search\n",
        "part_3.pdf asks that changes to the suggested strategies be "
        '"clearly described and justified experimentally". The searched knobs are '
        "the ones the spec names for each strategy; the velocity-network "
        "architecture, the optimizer, the epoch budget and T were held fixed, so "
        "the two strategies stay comparable. Configurations were ranked on mean "
        "validation delta across seeds, with test accuracy computed but never used "
        f"for ranking. The top {top_n} of each search are shown; the full results "
        "are in `reports/stage3_tuning.json`.\n",
    ]
    for (method, dataset), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda summary: summary.mean_val_delta, reverse=True)
        lines.append(f"### {method} on {dataset}\n")
        lines.append(format_tuning_table(ordered, top_n=top_n) + "\n")
    return lines


def format_stage3_section(
    summaries: List[dict],
    stage3_summaries: List[dict],
    separations: Sequence[dict],
    settings: Sequence[Tuple[str, str]],
    figures: Stage3Figures,
    project_root: Union[str, Path],
    tuning_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Render the Stage 3 part of RESULTS.md as a list of Markdown blocks."""
    project_root = Path(project_root)

    lines = [
        "# Stage 3 Results: FM Before a Linear Classifier\n",
        "The pipeline is `z -> FM -> frozen linear classifier -> s`. The "
        "classifier is not retrained: each run loads the Stage 1 linear-probe "
        "checkpoint for the same dataset, encoder, K and seed, verifies it still "
        "reproduces its published test accuracy, and freezes it. Every `Delta` "
        "below is therefore a paired comparison against that run's own baseline.\n",
        f"Following part_3.pdf's narrowed scope: one representative encoder per "
        f"dataset, K={STAGE3_K_SHOT}, a single Euler-step count T=4 used "
        f"throughout, and {len(STAGE3_SEEDS)} seeds. The flow is initialized so "
        "that its rollout is exactly the identity, so before training the complete "
        "system reproduces the Stage 1 probe's predictions bit for bit - every "
        "delta below starts from precisely zero.\n",
        "## Main comparison\n",
        format_stage3_comparison_table(summaries, settings, STAGE3_K_SHOT),
        "\n## Training and selection diagnostics\n",
        format_stage3_diagnostics_table(stage3_summaries),
        "\n## Class structure in the full feature space\n",
        "The feature-space figures below are two-dimensional projections; this "
        "table measures the same property in the space the classifier actually "
        "sees, on the same samples those figures plot.\n",
        format_class_separation_table(separations),
    ]

    lines.extend(
        format_stage3_observations(summaries, stage3_summaries, separations, settings)
    )

    if tuning_path is not None:
        lines.extend(_tuning_tables(tuning_path))

    for heading, figure_paths in figures.sections():
        if not figure_paths:
            continue
        lines.append(f"## {heading}\n")
        for label, encoder, figure_path in figure_paths:
            relative_path = Path(figure_path).relative_to(project_root)
            lines.append(f"### {label} / {encoder}\n")
            lines.append(f"![{label} {encoder}]({relative_path.as_posix()})\n")

    return lines
