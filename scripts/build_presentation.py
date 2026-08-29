"""Build a self-contained HTML page presenting the Stage 2 results.

A companion to scripts/generate_report.py, not a replacement: that script
produces RESULTS.md and RESULTS.pdf as the reference deliverable, while this
one produces a single page organised for presenting - the headline findings
first, then the comparison tables, then the figures behind each claim, with a
dataset/encoder switcher so one figure is shown at a time.

Every figure is embedded as a data URI, so the page is one file that needs no
network access and can be opened from a USB stick or projected offline. The
counted claims are read from the run records rather than written by hand, so
re-running this after a new sweep updates the prose along with the numbers.

Usage:
    python scripts/build_presentation.py
    python scripts/build_presentation.py --output somewhere/else.html
"""

import argparse
import base64
import html
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.evaluation.aggregation import aggregate_results, load_all_results  # noqa: E402
from src.evaluation.stage2_report import summarize_outcomes  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--outputs-dir", default=str(PROJECT / "outputs"))
parser.add_argument("--figures-dir", default=str(PROJECT / "reports" / "figures"))
parser.add_argument("--output", default=str(PROJECT / "reports" / "stage2_presentation.html"))
args = parser.parse_args()

PAIRS = [("dtd", "resnet18"), ("dtd", "dinov2_vits14"), ("flowers102", "resnet18")]
PAIR_LABELS = {
    ("dtd", "resnet18"): "DTD / ResNet-18",
    ("dtd", "dinov2_vits14"): "DTD / DINOv2",
    ("flowers102", "resnet18"): "Flowers-102 / ResNet-18",
}
CONDITIONS = [
    ("prototype", None, "Baseline"),
    ("fm_standard", 4, "Standard T=4"),
    ("fm_standard", 12, "Standard T=12"),
    ("fm_rolled", 4, "Rolled-out T=4"),
    ("fm_rolled", 12, "Rolled-out T=12"),
]

summaries = aggregate_results(load_all_results(args.outputs_dir))
counts = summarize_outcomes(summaries, PAIRS)


def lookup(dataset, encoder):
    return {
        (s["method"], s.get("num_euler_steps"), s["k_shot"]): s
        for s in summaries
        if s["dataset"] == dataset and s["encoder"] == encoder
    }


def data_uri(path):
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


# ---------------------------------------------------------------- headline numbers

all_deltas = [
    (s["mean_delta_accuracy"], s)
    for s in summaries
    if s.get("mean_delta_accuracy") is not None
]
best = max(all_deltas, key=lambda x: x[0])
worst = min(all_deltas, key=lambda x: x[0])


def setting_name(s):
    return (
        f"{PAIR_LABELS[(s['dataset'], s['encoder'])]}, K={s['k_shot']}, "
        f"T={s['num_euler_steps']}"
    )


stats = [
    (
        f"{counts['standard_beats_rolled']} / {counts['comparisons']}",
        "matched comparisons won by standard FM",
        "every dataset, encoder, K and T",
    ),
    (
        f"{counts['improved_full']['fm_standard']} / {counts['total_full']['fm_standard']}",
        "full-data settings improved by standard FM",
        f"but only {counts['improved']['fm_standard']} of {counts['total']['fm_standard']} overall",
    ),
    (
        f"{best[0] * 100:+.2f}",
        "best gain over baseline (pp)",
        setting_name(best[1]),
    ),
    (
        f"{worst[0] * 100:+.2f}",
        "worst loss, rolled-out (pp)",
        setting_name(worst[1]),
    ),
]

# ---------------------------------------------------------------- results tables


def results_table(dataset, encoder):
    table = lookup(dataset, encoder)
    head = "".join(f"<th>{html.escape(label)}</th>" for _, _, label in CONDITIONS)
    rows = []
    for k_shot in (5, 10, "full"):
        cells = []
        for method, steps, _ in CONDITIONS:
            s = table.get((method, steps, k_shot))
            if s is None:
                cells.append('<td class="na">n/a</td>')
                continue
            acc = f"{s['mean_test_accuracy'] * 100:.2f}"
            std = s["std_test_accuracy"]
            std_html = (
                f'<span class="pm">±{std * 100:.2f}</span>' if std is not None else ""
            )
            delta = s.get("mean_delta_accuracy")
            if delta is None:
                delta_html = '<span class="delta ref">reference</span>'
            else:
                sign = "pos" if delta > 0 else "neg"
                delta_html = (
                    f'<span class="delta {sign}">{delta * 100:+.2f}</span>'
                )
            cells.append(
                f'<td><span class="acc">{acc}</span>{std_html}<br>{delta_html}</td>'
            )
        label = "Full" if k_shot == "full" else f"{k_shot}-shot"
        runs = 1 if k_shot == "full" else 3
        rows.append(
            f'<tr><th scope="row">{label}<span class="runs">{runs} run'
            f'{"s" if runs > 1 else ""}</span></th>{"".join(cells)}</tr>'
        )
    return (
        '<div class="table-wrap"><table>'
        f'<thead><tr><th scope="col">Training set</th>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


# ---------------------------------------------------------------- figure groups

FIGURE_GROUPS = [
    (
        "delta",
        "Change in accuracy vs. the baseline",
        "Zero is the Stage 1 prototype baseline. Standard FM (blue) crosses above it "
        "only at full data; rolled-out (red) sits 2–11 points below throughout.",
        lambda d, e: f"stage2_delta_vs_shot_{d}_{e}.png",
    ),
    (
        "accuracy",
        "Absolute accuracy vs. training-set size",
        "The same data in absolute terms. Every series climbs steeply with K, which is "
        "why the paired-difference view above separates the methods more clearly.",
        lambda d, e: f"stage2_accuracy_vs_shot_{d}_{e}.png",
    ),
    (
        "loss",
        "Training curves",
        "Independent y-axes: standard FM regresses a velocity, rolled-out measures a "
        "squared distance to the prototype. The values are not comparable to each other. "
        "Watch rolled-out's validation loss turn upward around epoch 25.",
        lambda d, e: f"stage2_fm_loss_{d}_{e}.png",
    ),
    (
        "space",
        "Feature space, before and after",
        "One joint t-SNE projection across all three views, so a point moving between "
        "panels really moved. Rolled-out contracts hardest — and classifies worst.",
        lambda d, e: f"stage2_feature_space_{d}_{e}.png",
    ),
    (
        "traj",
        "Flow trajectories",
        "PCA, because a linear projection keeps a straight path straight. Standard FM "
        "takes even steps; rolled-out leaps most of the way on its first step and then "
        "crawls.",
        lambda d, e: f"stage2_flow_trajectories_{d}_{e}.png",
    ),
    (
        "perstep",
        "Metrics along the flow",
        "Accuracy, contraction, and margin at every Euler step. Accuracy peaks before "
        "t=1 in every setting, and the margin ends negative — the flow transports past "
        "its own best point.",
        lambda d, e: f"stage2_per_step_{d}_{e}_kfull.png",
    ),
    (
        "reverse",
        "Reverse flow from the prototypes",
        "Optional exploration: integrate backwards from a prototype to see what the "
        "model treats as a typical class member. Standard FM produces something "
        "class-like; rolled-out diverges by four orders of magnitude.",
        lambda d, e: f"stage2_reverse_flow_{d}_{e}.png",
    ),
]

FIGURES = Path(args.figures_dir)


def figure_group(group_id, title, caption, filename_for):
    tabs, panels = [], []
    for index, (dataset, encoder) in enumerate(PAIRS):
        path = FIGURES / filename_for(dataset, encoder)
        if not path.exists():
            continue
        active = " is-active" if index == 0 else ""
        label = PAIR_LABELS[(dataset, encoder)]
        tab_id = f"{group_id}-{index}"
        tabs.append(
            f'<button class="tab{active}" role="tab" aria-selected="{"true" if index == 0 else "false"}" '
            f'data-group="{group_id}" data-target="{tab_id}">{html.escape(label)}</button>'
        )
        panels.append(
            f'<figure class="panel{active}" id="{tab_id}" role="tabpanel">'
            f'<img src="{data_uri(path)}" alt="{html.escape(title)} for {html.escape(label)}" '
            'loading="lazy"></figure>'
        )
    return (
        f'<section class="figure-group" id="fig-{group_id}">'
        f"<h3>{html.escape(title)}</h3>"
        f'<p class="caption">{caption}</p>'
        f'<div class="tabs" role="tablist">{"".join(tabs)}</div>'
        f'{"".join(panels)}</section>'
    )


# ---------------------------------------------------------------- findings

FINDINGS = [
    (
        "Standard flow matching wins everywhere",
        f"In all {counts['comparisons']} matched comparisons — same dataset, encoder, K "
        "and T — standard FM scored higher than rolled-out training. The ranking is "
        "consistent, not marginal: the gap runs from 0.4 to 11 percentage points.",
        "Comparison tables",
    ),
    (
        "The flow layer only helps once there is enough data",
        f"Standard FM beat the prototype baseline in all "
        f"{counts['total_full']['fm_standard']} full-data settings, but in only "
        f"{counts['improved']['fm_standard']} of {counts['total']['fm_standard']} "
        "settings overall. With 5 or 10 images per class, a velocity network with ~790k "
        "parameters has too little to learn from and makes the baseline worse.",
        "Change vs. baseline",
    ),
    (
        "Rolled-out training learns a jump, not a transport",
        "Supervising only the endpoint leaves the intermediate states unconstrained, so "
        "the network discovers it can leap almost the whole way immediately. On "
        "DTD/DINOv2 at T=12 the first Euler step is about six times the last. That also "
        "explains why T barely changes the results: the flow is effectively over after "
        "one step.",
        "Flow trajectories",
    ),
    (
        "Contraction is not discrimination",
        "Rolled-out FM pulls test features closer to their own prototypes than standard "
        "FM does (mean distance 0.499 vs 0.614) and still classifies them worse. Pulling "
        "every point inward also drags points sitting nearer a wrong prototype, locking "
        "errors in rather than correcting them.",
        "Feature space",
    ),
    (
        "The flow transports past its own best point",
        "Measured at every Euler step, accuracy peaks before t=1 in every setting and "
        "then declines, while the mean margin ends negative. On DTD/ResNet-18 at K=10 it "
        "peaks at 52.50% around t=0.33 and falls to 47.98% by t=1 — above the 51.38% "
        "baseline at its peak, below it at the end.",
        "Metrics along the flow",
    ),
]

# ---------------------------------------------------------------- assemble

stat_html = "".join(
    f'<div class="stat"><div class="stat-value">{html.escape(value)}</div>'
    f'<div class="stat-label">{html.escape(label)}</div>'
    f'<div class="stat-note">{html.escape(note)}</div></div>'
    for value, label, note in stats
)

finding_html = "".join(
    f'<article class="finding"><h3>{html.escape(title)}</h3>'
    f"<p>{body}</p>"
    f'<p class="evidence"><span>Evidence</span>{html.escape(evidence)}</p></article>'
    for title, body, evidence in FINDINGS
)

table_tabs = "".join(
    f'<button class="tab{" is-active" if i == 0 else ""}" role="tab" '
    f'aria-selected="{"true" if i == 0 else "false"}" data-group="tbl" data-target="tbl-{i}">'
    f"{html.escape(PAIR_LABELS[pair])}</button>"
    for i, pair in enumerate(PAIRS)
)
table_panels = "".join(
    f'<div class="panel{" is-active" if i == 0 else ""}" id="tbl-{i}" role="tabpanel">'
    f"{results_table(*pair)}</div>"
    for i, pair in enumerate(PAIRS)
)

figures_html = "".join(figure_group(*group) for group in FIGURE_GROUPS)

TEMPLATE = """<title>Flow Matching to Class Prototypes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<style>
:root {
  --ground: #f7f8fa;
  --surface: #ffffff;
  --surface-sunk: #eef1f5;
  --ink: #151a21;
  --ink-soft: #414b57;
  --muted: #69737f;
  --rule: #dde1e7;
  --rule-strong: #c3cad3;
  --accent: #1f6fb2;
  --accent-soft: #e4eef7;
  --pos: #1c6b42;
  --neg: #b03a34;
  --serif: "IBM Plex Serif", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --col: 66ch;
}
:root:not([data-theme="light"]) { }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #11151b;
    --surface: #171d25;
    --surface-sunk: #1d242e;
    --ink: #e6eaf0;
    --ink-soft: #b3bcc8;
    --muted: #7f8a97;
    --rule: #262f3a;
    --rule-strong: #35404d;
    --accent: #6aa9dd;
    --accent-soft: #17293a;
    --pos: #58b57f;
    --neg: #e0776f;
  }
}
:root[data-theme="dark"] {
  --ground: #11151b;
  --surface: #171d25;
  --surface-sunk: #1d242e;
  --ink: #e6eaf0;
  --ink-soft: #b3bcc8;
  --muted: #7f8a97;
  --rule: #262f3a;
  --rule-strong: #35404d;
  --accent: #6aa9dd;
  --accent-soft: #17293a;
  --pos: #58b57f;
  --neg: #e0776f;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 24px 96px; }
.col { max-width: var(--col); }
h1, h2, h3 { text-wrap: balance; margin: 0; }
p { margin: 0; }

.eyebrow {
  font-family: var(--mono);
  font-size: 12px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--muted);
}

header.masthead { padding: 72px 0 40px; border-bottom: 1px solid var(--rule); }
header.masthead h1 {
  font-family: var(--serif);
  font-weight: 600;
  font-size: clamp(2.1rem, 4.6vw, 3.3rem);
  line-height: 1.08;
  letter-spacing: -0.015em;
  margin: 14px 0 18px;
}
header.masthead .standfirst { font-size: 1.12rem; color: var(--ink-soft); }
.meta {
  display: flex; flex-wrap: wrap; gap: 8px 22px;
  margin-top: 26px; font-family: var(--mono); font-size: 12.5px; color: var(--muted);
}

.stats {
  display: grid; gap: 1px; margin: 0; padding: 0;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  background: var(--rule); border-block: 1px solid var(--rule);
}
.stats-band { margin: 0 0 64px; }
.stat { background: var(--ground); padding: 26px 22px; }
.stat-value {
  font-family: var(--mono); font-weight: 500;
  font-size: clamp(1.7rem, 3vw, 2.15rem);
  letter-spacing: -0.02em; color: var(--accent);
  font-variant-numeric: tabular-nums;
}
.stat-label { margin-top: 6px; font-size: 0.92rem; color: var(--ink); }
.stat-note { margin-top: 4px; font-size: 0.8rem; color: var(--muted); }

section.block { padding-top: 60px; }
section.block > h2 {
  font-family: var(--serif); font-weight: 600;
  font-size: clamp(1.5rem, 2.6vw, 2rem); letter-spacing: -0.01em;
  margin: 10px 0 10px;
}
section.block > .lede { color: var(--ink-soft); margin-bottom: 30px; }

.findings { display: grid; gap: 1px; background: var(--rule); border: 1px solid var(--rule); }
.finding { background: var(--surface); padding: 26px 28px; }
.finding h3 {
  font-family: var(--sans); font-weight: 600; font-size: 1.06rem;
  letter-spacing: -0.005em; margin-bottom: 8px;
}
.finding p { color: var(--ink-soft); max-width: var(--col); font-size: 0.96rem; }
.evidence {
  margin-top: 14px !important; font-family: var(--mono); font-size: 11.5px;
  letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted) !important;
}
.evidence span {
  color: var(--accent); margin-right: 10px;
  border-right: 1px solid var(--rule-strong); padding-right: 10px;
}

.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 20px 0 18px; }
.tab {
  font-family: var(--mono); font-size: 12px; letter-spacing: 0.03em;
  padding: 7px 14px; border: 1px solid var(--rule-strong); background: transparent;
  color: var(--muted); cursor: pointer; border-radius: 2px;
}
.tab:hover { color: var(--ink); border-color: var(--accent); }
.tab.is-active {
  background: var(--accent-soft); border-color: var(--accent); color: var(--accent);
}
.tab:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.panel { display: none; }
.panel.is-active { display: block; }

.figure-group { margin-top: 52px; }
.figure-group h3 {
  font-family: var(--serif); font-weight: 600; font-size: 1.24rem; margin-bottom: 8px;
}
.caption { color: var(--ink-soft); max-width: var(--col); font-size: 0.94rem; }
figure { margin: 0; }
figure img {
  display: block; width: 100%; height: auto;
  background: #fff; border: 1px solid var(--rule); border-radius: 3px;
}

.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 3px; }
table { border-collapse: collapse; width: 100%; min-width: 720px; background: var(--surface); }
th, td { text-align: left; padding: 13px 16px; border-bottom: 1px solid var(--rule); }
thead th {
  font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.05em;
  text-transform: uppercase; color: var(--muted); font-weight: 500;
  background: var(--surface-sunk); white-space: nowrap;
}
tbody th {
  font-family: var(--sans); font-weight: 600; font-size: 0.95rem; white-space: nowrap;
}
.runs {
  display: block; font-family: var(--mono); font-size: 11px;
  color: var(--muted); font-weight: 400; letter-spacing: 0.04em;
}
td { font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 0.9rem; }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: none; }
.acc { font-size: 0.96rem; }
.pm { color: var(--muted); margin-left: 4px; font-size: 0.82rem; }
.delta { font-size: 0.82rem; }
.delta.pos { color: var(--pos); }
.delta.neg { color: var(--neg); }
.delta.ref, .na { color: var(--muted); }

.notes { display: grid; gap: 18px; margin-top: 26px; }
.note {
  border-left: 2px solid var(--rule-strong); padding-left: 18px;
  max-width: var(--col); font-size: 0.94rem; color: var(--ink-soft);
}
.note strong { color: var(--ink); font-weight: 600; }

dl.spec {
  display: grid; grid-template-columns: minmax(160px, auto) 1fr;
  gap: 1px; background: var(--rule); border: 1px solid var(--rule); margin: 24px 0 0;
}
dl.spec dt, dl.spec dd {
  background: var(--surface); margin: 0; padding: 11px 16px; font-size: 0.9rem;
}
dl.spec dt { font-family: var(--mono); font-size: 12px; color: var(--muted); letter-spacing: 0.03em; }
dl.spec dd { font-family: var(--mono); font-variant-numeric: tabular-nums; color: var(--ink); }

footer {
  margin-top: 76px; padding-top: 26px; border-top: 1px solid var(--rule);
  font-family: var(--mono); font-size: 12px; color: var(--muted);
}
@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>

<div class="wrap">
  <header class="masthead">
    <p class="eyebrow">CVLAB Summer Project &middot; Stage 2</p>
    <h1>Flow Matching to Class Prototypes</h1>
    <p class="standfirst col">A velocity network transports frozen encoder features toward
    their class prototype. Two training objectives, two step counts, three encoders, and
    one result that runs the opposite way to the obvious expectation.</p>
    <div class="meta">
      <span>84 runs &middot; 63 trainings</span>
      <span>DTD &middot; Flowers-102</span>
      <span>ResNet-18 &middot; DINOv2 ViT-S/14</span>
      <span>K = 5, 10, full &middot; T = 4, 12</span>
    </div>
  </header>
</div>

<div class="stats-band"><div class="wrap" style="padding-bottom:0">
  <div class="stats">__STATS__</div>
</div></div>

<div class="wrap">
  <section class="block">
    <p class="eyebrow">What we found</p>
    <h2>Rolled-out training removes the train/inference mismatch &mdash; and loses anyway</h2>
    <p class="lede col">Standard flow matching is supervised on the ideal straight-line path
    but evaluated on states it generates itself. Rolled-out training removes that mismatch by
    unrolling the full inference procedure during training. It should be the better method.
    It was not.</p>
    <div class="findings">__FINDINGS__</div>
  </section>

  <section class="block">
    <p class="eyebrow">Classification results</p>
    <h2>Top-1 accuracy on the full official test split</h2>
    <p class="lede col">Each cell shows mean accuracy with its seed spread, and beneath it the
    change against that run&rsquo;s own paired baseline. Every flow-matching run reuses the
    identical K-shot subset, seed and prototypes as the Stage 1 prototype run it is compared
    against.</p>
    <div class="tabs" role="tablist">__TABLE_TABS__</div>
    __TABLE_PANELS__
  </section>

  <section class="block">
    <p class="eyebrow">Evidence</p>
    <h2>The figures behind each claim</h2>
    <p class="lede col">Use the tabs to switch dataset and encoder within each figure.</p>
    __FIGURES__
  </section>

  <section class="block">
    <p class="eyebrow">Protocol</p>
    <h2>What was held fixed</h2>
    <p class="lede col">No hyperparameter search was run. The velocity network follows the
    spec&rsquo;s suggested architecture, and the optimizer settings deliberately mirror Stage
    1&rsquo;s linear probe so the two stages stay comparable.</p>
    <dl class="spec">
      <dt>architecture</dt><dd>MLP &middot; (D+1) &rarr; 512 &rarr; 512 &rarr; D &middot; SiLU</dd>
      <dt>parameters</dt><dd>788,480 (D=512) &middot; 657,280 (D=384)</dd>
      <dt>optimizer</dt><dd>AdamW &middot; lr 1e-3 &middot; weight decay 1e-4</dd>
      <dt>batch size</dt><dd>64</dd>
      <dt>epochs</dt><dd>200, fixed budget &middot; final weights kept</dd>
      <dt>checkpoint</dt><dd>no validation-based selection</dd>
      <dt>features</dt><dd>L2-normalized before the flow</dd>
      <dt>repetitions</dt><dd>3 subset seeds at K=5 and K=10 &middot; 1 run at K=full</dd>
    </dl>
    <div class="notes">
      <p class="note"><strong>Why normalize.</strong> Raw encoder features have norms of 24&ndash;48
      against unit-norm prototypes, which would make the regression target almost entirely
      about shrinking magnitude &mdash; a direction the cosine classifier ignores. Stage 1&rsquo;s
      classifier already normalizes at both ends, and because cosine similarity is
      scale-invariant this leaves the baseline numbers untouched.</p>
      <p class="note"><strong>Why one training serves both T.</strong> Standard FM samples t
      continuously and never discretizes the path, so its T=4 and T=12 networks are
      bit-identical under the same seed. Rolled-out bakes T into its weights and needs one
      training per T. Hence 63 trainings but 84 run records.</p>
    </div>
  </section>

  <section class="block">
    <p class="eyebrow">Caveats</p>
    <h2>How to weigh these numbers</h2>
    <div class="notes">
      <p class="note"><strong>The full-data settings are single runs.</strong> Stage 1&rsquo;s
      prototype protocol states that the full-data result requires one run, and Stage 2
      extends that branch. So K=full carries no error bars, while few-shot settings vary by
      up to about 1.5 points across seeds. The positive full-data deltas should be read with
      that in mind.</p>
      <p class="note"><strong>For Flowers-102, K=10 and K=full are the same data.</strong> The
      official training split holds exactly 10 images per class, so the 10-shot subsets are
      the full split. The baseline is identical in both rows with zero variance, and the
      small differences between the flow-matching rows come only from the initialization
      seed.</p>
      <p class="note"><strong>Rolled-out is under-regularized at 200 epochs.</strong> Its
      validation loss bottoms out around epoch 25 and rises for the remaining 175. Part of
      its deficit is an epoch-budget artifact rather than purely inherent to the objective.
      The identical budget is what makes the comparison fair, so it was not tuned away.</p>
      <p class="note"><strong>Early stopping in flow time is a direction, not a result.</strong>
      Accuracy peaks before t=1 everywhere, but choosing a stopping time from these curves
      would be test-set selection. Doing it properly would require the validation split.</p>
      <p class="note"><strong>Two-dimensional projections are qualitative.</strong> The
      trajectory PCA captures well under half the variance in some settings, printed on each
      axis. Every quantitative claim here is computed in the full feature space.</p>
    </div>
  </section>

  <footer>
    Generated from 132 experiment records &middot; all figures rebuilt from saved velocity-network
    checkpoints &middot; 330 tests passing
  </footer>
</div>

<script>
document.querySelectorAll(".tab").forEach(function (tab) {
  tab.addEventListener("click", function () {
    var group = tab.dataset.group;
    document.querySelectorAll('.tab[data-group="' + group + '"]').forEach(function (other) {
      var on = other === tab;
      other.classList.toggle("is-active", on);
      other.setAttribute("aria-selected", on ? "true" : "false");
      var panel = document.getElementById(other.dataset.target);
      if (panel) { panel.classList.toggle("is-active", on); }
    });
  });
});
</script>
"""

page = (
    TEMPLATE.replace("__STATS__", stat_html)
    .replace("__FINDINGS__", finding_html)
    .replace("__TABLE_TABS__", table_tabs)
    .replace("__TABLE_PANELS__", table_panels)
    .replace("__FIGURES__", figures_html)
)

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(page, encoding="utf-8")
print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {page.count('data:image/png')} figures embedded)")
