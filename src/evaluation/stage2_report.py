"""Build every Stage 2 figure and the Markdown section that presents them.

Factored out of scripts/generate_report.py so the Stage 2 reporting logic can
be unit-tested against a tiny synthetic setup, and so the report script stays
a thin orchestrator rather than several hundred lines of figure plumbing.

Each figure needs a representative run to draw. Those choices are project
conventions, recorded as module constants rather than buried in the code:

  - loss curves use the 10-shot, seed 0 run, matching the convention Stage 1
    already uses for its linear-probe curves;
  - the feature-space, trajectory and reverse-flow figures use the K=full run
    at T=12 - the setting where the FM layer actually helps, so the geometry
    shown is that of a working transport rather than a failing one;
  - the per-step curves are drawn for both 10-shot and full, because the
    contrast between them is the point: the accuracy peak along the flow sits
    at a different time in each.

Anything missing from outputs/ is skipped with a message rather than raising,
so a partially-completed sweep still produces a report.
"""

import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import torch

from src.data.datasets import get_class_names, load_dataset_splits
from src.evaluation.tables import format_flow_matching_comparison_table
from src.flow_matching.inference import (
    reverse_trajectory_with_checkpoint,
    trajectory_with_checkpoint,
    transport_with_checkpoint,
)
from src.flow_matching.per_step import compute_per_step_metrics
from src.flow_matching.runner import baseline_accuracy, flow_matching_run_dir, prepare_features
from src.utils.config import ExperimentConfig, load_config
from src.visualization.accuracy_vs_shot import plot_accuracy_vs_shot, plot_delta_vs_shot
from src.visualization.feature_space import (
    load_selection,
    plot_feature_space_comparison,
    project_feature_groups,
    save_selection,
    select_classes_and_samples,
)
from src.visualization.flow_trajectories import (
    plot_flow_trajectories,
    project_trajectories,
    select_trajectory_examples,
)
from src.visualization.loss_curves import load_history, plot_flow_matching_loss_curves
from src.visualization.per_step_curves import plot_per_step_curves

LOSS_CURVE_K_SHOT = 10
LOSS_CURVE_SEED = 0

GEOMETRY_K_SHOT = "full"
GEOMETRY_SEED = 0
GEOMETRY_EULER_STEPS = 12

TRAJECTORY_NUM_CLASSES = 5
TRAJECTORY_EXAMPLES_PER_CLASS = 2

PER_STEP_K_SHOTS = (10, "full")
PER_STEP_SEED = 0

FM_METHODS = ("fm_standard", "fm_rolled")
EULER_STEP_COUNTS = (4, 12)

# (dataset, encoder, absolute path) - the shape the report and PDF consume.
FigureList = List[Tuple[str, str, Path]]


@dataclasses.dataclass
class Stage2Figures:
    """Every Stage 2 figure produced, grouped by the section it belongs to."""

    accuracy_vs_shot: FigureList = dataclasses.field(default_factory=list)
    delta_vs_shot: FigureList = dataclasses.field(default_factory=list)
    loss_curves: FigureList = dataclasses.field(default_factory=list)
    feature_space: FigureList = dataclasses.field(default_factory=list)
    trajectories: FigureList = dataclasses.field(default_factory=list)
    reverse_flow: FigureList = dataclasses.field(default_factory=list)
    per_step: FigureList = dataclasses.field(default_factory=list)

    def sections(self) -> List[Tuple[str, FigureList]]:
        """(heading, figures) in the order the report presents them."""
        return [
            ("Stage 2: accuracy vs. training-set size", self.accuracy_vs_shot),
            ("Stage 2: change in accuracy relative to the prototype baseline", self.delta_vs_shot),
            ("Stage 2: flow-matching training curves (10-shot, seed 0)", self.loss_curves),
            ("Stage 2: feature space before and after flow matching (t-SNE)", self.feature_space),
            ("Stage 2: flow trajectories (PCA)", self.trajectories),
            ("Stage 2: reverse flow from the class prototypes (PCA)", self.reverse_flow),
            ("Stage 2: metrics along the flow", self.per_step),
        ]


def _run_dir(output_dir, dataset, encoder, method, k_shot, num_euler_steps, seed) -> Path:
    return flow_matching_run_dir(
        output_dir, dataset, encoder, method, k_shot, num_euler_steps, seed
    )


def _load_checkpoint(run_dir: Path) -> Optional[Tuple[dict, Sequence[int]]]:
    """Load a run's weights and the hidden widths they were trained with.

    The widths come from the run's own saved config rather than a constant,
    so a report built from older runs still rebuilds the right network.
    """
    checkpoint_path = run_dir / "checkpoint.pt"
    config_path = run_dir / "config.yaml"
    if not checkpoint_path.exists() or not config_path.exists():
        return None
    state_dict = torch.load(checkpoint_path, weights_only=True)
    hidden_dims = load_config(config_path).flow_matching.hidden_dims
    return state_dict, hidden_dims


def _prepared(dataset, encoder, k_shot, seed, cache_dir):
    config = ExperimentConfig(
        dataset=dataset, encoder=encoder, method="fm_standard", k_shot=k_shot, seed=seed
    )
    return prepare_features(config, cache_dir)


def _plot_accuracy_figures(summaries, dataset, encoder, figures_dir) -> Tuple[Optional[Path], Optional[Path]]:
    accuracy_path = figures_dir / f"stage2_accuracy_vs_shot_{dataset}_{encoder}.png"
    delta_path = figures_dir / f"stage2_delta_vs_shot_{dataset}_{encoder}.png"
    try:
        plot_accuracy_vs_shot(
            summaries, dataset, encoder, accuracy_path,
            methods=["prototype", *FM_METHODS],
            title=f"{dataset} / {encoder}: prototype baseline vs. flow matching",
        )
        plot_delta_vs_shot(summaries, dataset, encoder, delta_path)
    except ValueError as error:
        print(f"  Skipping Stage 2 accuracy plots for {dataset}/{encoder}: {error}")
        return None, None
    return accuracy_path, delta_path


def _plot_loss_curves(dataset, encoder, output_dir, figures_dir) -> Optional[Path]:
    standard_path = (
        _run_dir(output_dir, dataset, encoder, "fm_standard", LOSS_CURVE_K_SHOT, 4, LOSS_CURVE_SEED)
        / "history.json"
    )
    if not standard_path.exists():
        print(f"  Skipping Stage 2 loss curves for {dataset}/{encoder}: no run at {standard_path}")
        return None

    rolled_histories = {}
    for num_euler_steps in EULER_STEP_COUNTS:
        history_path = (
            _run_dir(
                output_dir, dataset, encoder, "fm_rolled",
                LOSS_CURVE_K_SHOT, num_euler_steps, LOSS_CURVE_SEED,
            )
            / "history.json"
        )
        if history_path.exists():
            rolled_histories[num_euler_steps] = load_history(history_path)
    if not rolled_histories:
        print(f"  Skipping Stage 2 loss curves for {dataset}/{encoder}: no rolled-out runs")
        return None

    save_path = figures_dir / f"stage2_fm_loss_{dataset}_{encoder}.png"
    plot_flow_matching_loss_curves(
        load_history(standard_path), rolled_histories, dataset, encoder,
        LOSS_CURVE_K_SHOT, LOSS_CURVE_SEED, save_path,
    )
    return save_path


def _plot_feature_space(dataset, encoder, cache_dir, data_dir, output_dir, reports_dir, figures_dir, device) -> Optional[Path]:
    checkpoints = {}
    for method in FM_METHODS:
        loaded = _load_checkpoint(
            _run_dir(output_dir, dataset, encoder, method, GEOMETRY_K_SHOT, GEOMETRY_EULER_STEPS, GEOMETRY_SEED)
        )
        if loaded is None:
            print(f"  Skipping Stage 2 feature space for {dataset}/{encoder}: no {method} K={GEOMETRY_K_SHOT} run")
            return None
        checkpoints[method] = loaded

    data = _prepared(dataset, encoder, GEOMETRY_K_SHOT, GEOMETRY_SEED, cache_dir)

    # Reuse Stage 1's saved selection so the same classes, test examples and
    # colours appear in both stages' feature-space figures.
    selection_path = reports_dir / f"feature_viz_selection_{dataset}.json"
    if selection_path.exists():
        selection = load_selection(selection_path)
    else:
        selection = select_classes_and_samples(dataset, data.test_labels, seed=0)
        save_selection(selection, selection_path)

    original = data.test_features[selection.sample_indices]
    sample_class_ids = data.test_labels[selection.sample_indices].tolist()
    prototypes = data.prototypes[selection.class_ids]

    transported = {
        method: transport_with_checkpoint(
            state_dict, hidden_dims, original, GEOMETRY_EULER_STEPS, device
        )
        for method, (state_dict, hidden_dims) in checkpoints.items()
    }

    original_2d, standard_2d, rolled_2d, prototype_2d = project_feature_groups(
        [original, transported["fm_standard"], transported["fm_rolled"], prototypes], seed=0
    )

    class_names = get_class_names(load_dataset_splits(dataset, data_dir, download=False)["train"])
    save_path = figures_dir / f"stage2_feature_space_{dataset}_{encoder}.png"
    plot_feature_space_comparison(
        [
            ("original encoder features", original_2d),
            (f"after standard FM (T={GEOMETRY_EULER_STEPS})", standard_2d),
            (f"after rolled-out FM (T={GEOMETRY_EULER_STEPS})", rolled_2d),
        ],
        prototype_2d, sample_class_ids, selection.class_ids, class_names,
        f"{dataset} / {encoder}: feature space before and after flow matching "
        f"(K={GEOMETRY_K_SHOT}, seed {GEOMETRY_SEED}, {len(selection.class_ids)} classes)",
        save_path,
    )
    return save_path


def _plot_trajectories(dataset, encoder, cache_dir, data_dir, output_dir, reports_dir, figures_dir, device) -> Optional[Path]:
    checkpoints = {}
    for method in FM_METHODS:
        loaded = _load_checkpoint(
            _run_dir(output_dir, dataset, encoder, method, GEOMETRY_K_SHOT, GEOMETRY_EULER_STEPS, GEOMETRY_SEED)
        )
        if loaded is None:
            print(f"  Skipping Stage 2 trajectories for {dataset}/{encoder}: no {method} run")
            return None
        checkpoints[method] = loaded

    data = _prepared(dataset, encoder, GEOMETRY_K_SHOT, GEOMETRY_SEED, cache_dir)
    selection_path = reports_dir / f"feature_viz_selection_{dataset}.json"
    selection = (
        load_selection(selection_path)
        if selection_path.exists()
        else select_classes_and_samples(dataset, data.test_labels, seed=0)
    )
    class_ids = selection.class_ids[:TRAJECTORY_NUM_CLASSES]

    indices = select_trajectory_examples(
        data.test_labels, class_ids, TRAJECTORY_EXAMPLES_PER_CLASS, seed=0
    )
    start = data.test_features[indices]
    sample_class_ids = data.test_labels[indices].tolist()
    prototypes = data.prototypes[class_ids]

    trajectories = {
        method: trajectory_with_checkpoint(
            state_dict, hidden_dims, start, GEOMETRY_EULER_STEPS, device
        )
        for method, (state_dict, hidden_dims) in checkpoints.items()
    }
    (standard_2d, rolled_2d), prototype_2d, ratio = project_trajectories(
        [trajectories["fm_standard"], trajectories["fm_rolled"]], prototypes
    )

    class_names = get_class_names(load_dataset_splits(dataset, data_dir, download=False)["train"])
    save_path = figures_dir / f"stage2_flow_trajectories_{dataset}_{encoder}.png"
    plot_flow_trajectories(
        [
            (f"standard FM (T={GEOMETRY_EULER_STEPS})", standard_2d),
            (f"rolled-out FM (T={GEOMETRY_EULER_STEPS})", rolled_2d),
        ],
        prototype_2d, sample_class_ids, class_ids, class_names,
        f"{dataset} / {encoder}: Euler flow trajectories "
        f"(K={GEOMETRY_K_SHOT}, seed {GEOMETRY_SEED}, {len(indices)} test examples)",
        save_path, explained_variance_ratio=ratio,
    )
    return save_path


def _plot_reverse_flow(dataset, encoder, cache_dir, data_dir, output_dir, reports_dir, figures_dir, device) -> Optional[Path]:
    data = _prepared(dataset, encoder, GEOMETRY_K_SHOT, GEOMETRY_SEED, cache_dir)
    selection_path = reports_dir / f"feature_viz_selection_{dataset}.json"
    selection = (
        load_selection(selection_path)
        if selection_path.exists()
        else select_classes_and_samples(dataset, data.test_labels, seed=0)
    )
    class_ids = selection.class_ids[:TRAJECTORY_NUM_CLASSES]
    prototypes = data.prototypes[class_ids]

    # Real test features of the same classes, so the point a reverse flow
    # generates can be judged against the class it claims to reconstruct
    # rather than against an empty panel. stage_2.pdf's optional item asks to
    # compare *samples and prototypes*, so the samples have to be on the plot.
    keep = [i for i in selection.sample_indices if int(data.test_labels[i]) in set(class_ids)]
    sample_features = data.test_features[keep]
    sample_class_ids = data.test_labels[keep].tolist()

    panels, prototypes_per_panel, backgrounds = [], [], []
    for method, label in [("fm_standard", "standard FM"), ("fm_rolled", "rolled-out FM")]:
        loaded = _load_checkpoint(
            _run_dir(output_dir, dataset, encoder, method, GEOMETRY_K_SHOT, GEOMETRY_EULER_STEPS, GEOMETRY_SEED)
        )
        if loaded is None:
            print(f"  Skipping Stage 2 reverse flow for {dataset}/{encoder}: no {method} run")
            return None
        state_dict, hidden_dims = loaded
        trajectory = reverse_trajectory_with_checkpoint(
            state_dict, hidden_dims, prototypes, GEOMETRY_EULER_STEPS, device
        )
        # Each panel gets its own PCA: a diverging reverse field can be
        # several orders of magnitude larger than a well-behaved one, and a
        # shared basis would then be chosen entirely by the diverging panel.
        # The real samples join that fit (as a one-state "trajectory") so they
        # land in the same basis as the paths they are being compared against.
        (projected, projected_samples), projected_prototypes, ratio = project_trajectories(
            [trajectory, sample_features.unsqueeze(0)], prototypes
        )
        backgrounds.append((projected_samples[0], sample_class_ids))
        endpoint_norm = trajectory[-1].norm(dim=1).mean().item()
        panels.append(
            (
                f"{label} reversed  |  endpoint norm {endpoint_norm:.2f}  |  own PCA "
                f"{ratio.sum() * 100:.0f}% var",
                projected,
            )
        )
        prototypes_per_panel.append(projected_prototypes)

    class_names = get_class_names(load_dataset_splits(dataset, data_dir, download=False)["train"])
    save_path = figures_dir / f"stage2_reverse_flow_{dataset}_{encoder}.png"
    plot_flow_trajectories(
        panels, prototypes_per_panel, class_ids, class_ids, class_names,
        f"{dataset} / {encoder}: reverse flow from the class prototypes "
        f"(K={GEOMETRY_K_SHOT}, seed {GEOMETRY_SEED}, T={GEOMETRY_EULER_STEPS})"
        "\nfaint dots are real test features of the same classes - does the generated point land among them?"
        "\neach panel has its own PCA basis and scale - not comparable by eye; see endpoint norms",
        save_path, share_limits=False, background=backgrounds,
        legend_title=(
            "Class\n(faint dot = real test image, * = prototype / start,\n"
            ". = reverse step, X = generated point at t=0)"
        ),
    )
    return save_path


def _plot_per_step(dataset, encoder, k_shot, cache_dir, output_dir, figures_dir, device) -> Optional[Path]:
    data = _prepared(dataset, encoder, k_shot, PER_STEP_SEED, cache_dir)

    series = []
    for method in FM_METHODS:
        for num_euler_steps in EULER_STEP_COUNTS:
            loaded = _load_checkpoint(
                _run_dir(output_dir, dataset, encoder, method, k_shot, num_euler_steps, PER_STEP_SEED)
            )
            if loaded is None:
                continue
            state_dict, hidden_dims = loaded
            trajectory = trajectory_with_checkpoint(
                state_dict, hidden_dims, data.test_features, num_euler_steps, device
            )
            series.append(
                (method, num_euler_steps, compute_per_step_metrics(trajectory, data.test_labels, data.prototypes))
            )

    if not series:
        print(f"  Skipping Stage 2 per-step curves for {dataset}/{encoder} K={k_shot}: no runs")
        return None

    save_path = figures_dir / f"stage2_per_step_{dataset}_{encoder}_k{k_shot}.png"
    plot_per_step_curves(
        series, dataset, encoder, k_shot, save_path,
        baseline_accuracy=baseline_accuracy(
            data.test_features, data.test_labels, data.prototypes
        ),
    )
    return save_path


def generate_stage2_figures(
    summaries: List[dict],
    dataset_encoder_pairs: Sequence[Tuple[str, str]],
    cache_dir: Union[str, Path],
    data_dir: Union[str, Path],
    output_dir: Union[str, Path],
    reports_dir: Union[str, Path],
    figures_dir: Union[str, Path],
    device: torch.device,
) -> Stage2Figures:
    """Produce every Stage 2 figure for every dataset/encoder pair.

    Args:
        summaries: aggregated results from aggregation.aggregate_results().
        dataset_encoder_pairs: which pairs to draw.
        cache_dir, data_dir, output_dir, reports_dir, figures_dir: project directories.
        device: device to run the transports on.

    Returns:
        A `Stage2Figures` holding the paths actually written. Missing runs are
        skipped with a printed message, so a partial sweep still reports.
    """
    figures_dir = Path(figures_dir)
    reports_dir = Path(reports_dir)
    figures = Stage2Figures()

    for dataset, encoder in dataset_encoder_pairs:
        accuracy_path, delta_path = _plot_accuracy_figures(summaries, dataset, encoder, figures_dir)
        if accuracy_path:
            figures.accuracy_vs_shot.append((dataset, encoder, accuracy_path))
            figures.delta_vs_shot.append((dataset, encoder, delta_path))

        loss_path = _plot_loss_curves(dataset, encoder, output_dir, figures_dir)
        if loss_path:
            figures.loss_curves.append((dataset, encoder, loss_path))

        for builder, collection in [
            (_plot_feature_space, figures.feature_space),
            (_plot_trajectories, figures.trajectories),
            (_plot_reverse_flow, figures.reverse_flow),
        ]:
            path = builder(
                dataset, encoder, cache_dir, data_dir, output_dir, reports_dir, figures_dir, device
            )
            if path:
                collection.append((dataset, encoder, path))

        for k_shot in PER_STEP_K_SHOTS:
            path = _plot_per_step(dataset, encoder, k_shot, cache_dir, output_dir, figures_dir, device)
            if path:
                figures.per_step.append((dataset, f"{encoder} (K={k_shot})", path))

    return figures


def format_stage2_section(
    summaries: List[dict],
    dataset_encoder_pairs: Sequence[Tuple[str, str]],
    figures: Stage2Figures,
    project_root: Union[str, Path],
) -> List[str]:
    """Render the Stage 2 part of RESULTS.md as a list of Markdown blocks."""
    project_root = Path(project_root)
    lines = [
        "# Stage 2 Results: Flow Matching to Class Prototypes\n",
        "Each flow-matching run reuses the *same* K-shot subset, seed and class "
        "prototypes as the corresponding Stage 1 prototype run, so every "
        "`Delta` below is a paired comparison against that run's own baseline.\n",
        "## Comparison tables\n",
    ]
    for dataset, encoder in dataset_encoder_pairs:
        lines.append(f"### {dataset} / {encoder}\n")
        lines.append(format_flow_matching_comparison_table(summaries, dataset, encoder))

    lines.extend(format_observations(summaries, dataset_encoder_pairs))

    for heading, figure_paths in figures.sections():
        if not figure_paths:
            continue
        lines.append(f"## {heading}\n")
        for label, encoder, figure_path in figure_paths:
            relative_path = Path(figure_path).relative_to(project_root)
            lines.append(f"### {label} / {encoder}\n")
            lines.append(f"![{label} {encoder}]({relative_path.as_posix()})\n")

    return lines


def _setting_lookup(summaries, dataset, encoder):
    return {
        (s["method"], s.get("num_euler_steps"), s["k_shot"]): s
        for s in summaries
        if s["dataset"] == dataset and s["encoder"] == encoder
    }


def summarize_outcomes(
    summaries: List[dict], dataset_encoder_pairs: Sequence[Tuple[str, str]]
) -> dict:
    """Count the headline outcomes so the write-up states measured facts.

    Deriving these from the summaries rather than hardcoding them means the
    prose cannot drift out of step with the numbers if the sweep is re-run.

    Returns:
        A dict of counts: how often standard FM beat rolled-out, and how often
        each variant beat the prototype baseline, overall and restricted to
        the full-data settings.
    """
    standard_beats_rolled = 0
    comparisons = 0
    improved = {"fm_standard": 0, "fm_rolled": 0}
    total = {"fm_standard": 0, "fm_rolled": 0}
    improved_full = {"fm_standard": 0, "fm_rolled": 0}
    total_full = {"fm_standard": 0, "fm_rolled": 0}

    for dataset, encoder in dataset_encoder_pairs:
        lookup = _setting_lookup(summaries, dataset, encoder)
        for k_shot in (5, 10, "full"):
            for num_euler_steps in EULER_STEP_COUNTS:
                standard = lookup.get(("fm_standard", num_euler_steps, k_shot))
                rolled = lookup.get(("fm_rolled", num_euler_steps, k_shot))
                if standard and rolled:
                    comparisons += 1
                    if standard["mean_test_accuracy"] > rolled["mean_test_accuracy"]:
                        standard_beats_rolled += 1
                for method, summary in (("fm_standard", standard), ("fm_rolled", rolled)):
                    if summary is None or summary.get("mean_delta_accuracy") is None:
                        continue
                    total[method] += 1
                    if summary["mean_delta_accuracy"] > 0:
                        improved[method] += 1
                    if k_shot == "full":
                        total_full[method] += 1
                        if summary["mean_delta_accuracy"] > 0:
                            improved_full[method] += 1

    return {
        "standard_beats_rolled": standard_beats_rolled,
        "comparisons": comparisons,
        "improved": improved,
        "total": total,
        "improved_full": improved_full,
        "total_full": total_full,
    }


def format_observations(
    summaries: List[dict], dataset_encoder_pairs: Sequence[Tuple[str, str]]
) -> List[str]:
    """The discussion section: what the results show, and the caveats needed
    to weigh them.

    The counted claims are derived from the summaries so they cannot drift
    away from the tables; the mechanism claims describe what the figures
    show.
    """
    counts = summarize_outcomes(summaries, dataset_encoder_pairs)
    standard_improved = counts["improved"]["fm_standard"]
    rolled_improved = counts["improved"]["fm_rolled"]
    total = counts["total"]["fm_standard"]

    return [
        "## Observations\n",
        f"**Standard FM outperforms rolled-out training in "
        f"{counts['standard_beats_rolled']} of {counts['comparisons']} matched "
        "comparisons** (same dataset, encoder, K and T). The ranking is consistent "
        "rather than marginal.\n",
        f"**The flow-matching layer helps only when there is enough data.** Standard "
        f"FM improves on the prototype baseline in {standard_improved} of {total} "
        f"settings overall, but in {counts['improved_full']['fm_standard']} of the "
        f"{counts['total_full']['fm_standard']} full-data settings; rolled-out "
        f"improves in {rolled_improved} of {total}. The change relative to the "
        "baseline grows with training-set size on both DTD encoders (see the "
        "delta-vs-K plots).\n",
        "**Rolled-out training overfits its own objective.** Its final training loss "
        "is roughly an order of magnitude below standard FM's, yet its validation "
        "loss *rises* after roughly 25 epochs and keeps rising for the rest of the "
        "200-epoch budget (see the training curves). Supervising only the endpoint "
        "lets the network memorize the transport of its training points.\n",
        "**Rolled-out training learns a jump, not a transport.** Because nothing "
        "constrains the intermediate states, its first Euler step is several times "
        "larger than its last, so the flow effectively completes immediately. The "
        "trajectory plots show this directly, and it explains why T makes so little "
        "difference to the results.\n",
        "**Contraction is not discrimination.** Rolled-out FM pulls test features "
        "*closer* to their own prototypes than standard FM does and still classifies "
        "them worse: pulling every point inward also drags points that sit nearer a "
        "wrong prototype, locking in errors instead of correcting them. The per-step "
        "figures separate these two quantities for exactly this reason - similarity "
        "to the own prototype rises while the margin against the best competing "
        "prototype falls.\n",
        "**The flow over-transports.** In every setting measured, accuracy along the "
        "flow peaks *before* t=1 and then declines, and the mean margin ends "
        "negative. Integrating to completion overshoots the point of best class "
        "separation. This is the clearest direction for further work, but note that "
        "picking a stopping time from these curves would be test-set selection; "
        "doing it properly would require the validation split.\n",
        "**Reverse flow separates the two variants sharply.** Integrating backwards "
        "from a prototype, standard FM produces a point that genuinely resembles real "
        "members of that class - it lands inside the cloud of real test features, at "
        "positive cosine similarity to them. Rolled-out FM diverges by orders of "
        "magnitude and lands at *negative* similarity to every class, so it is not a "
        "plausible sample of anything; it still preserves the relative class ordering "
        "on the ResNet-18 pairs and loses even that on DINOv2. A one-step jump is not "
        "an invertible field.\n",
        "### Caveats\n",
        "- **The full-data settings are single runs.** Following the Stage 1 "
        "prototype protocol (\"the full-data result requires one run\"), K=full has no "
        "repetitions and therefore no error bars, while the few-shot settings vary by "
        "up to roughly 1.5 percentage points across seeds. The positive full-data "
        "deltas should be read with that in mind.\n",
        "- **For Flowers-102, K=10 and K=full are the same data.** The official "
        "training split holds exactly 10 images per class, so the 10-shot subsets are "
        "the full split. The prototype baseline is identical in both rows with zero "
        "variance, and the small differences between the FM rows come only from the "
        "velocity network's initialization seed.\n",
        "- **Two-dimensional projections are qualitative.** The trajectory PCA "
        "captures well under half the variance in some settings (printed on each "
        "axis), so apparent distances understate the true geometry. Every "
        "quantitative claim above is computed in the full feature space.\n",
    ]
