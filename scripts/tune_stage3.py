"""Run Stage 3's hyperparameter search and write the results.

part_3.pdf asks for the Stage 3 strategies' details to be experimented with
and any changes "clearly described and justified experimentally". This script
produces that justification: for each strategy and dataset it trains the full
grid over every seed, ranks configurations by mean validation delta, and
saves the whole search - not just the winner - to reports/stage3_tuning.json
for the report to render.

Test accuracy is computed for every configuration but never used to rank
them. It appears in the output so a reader can see how well the
validation-selected choice transferred.

Usage:
    python scripts/tune_stage3.py
    python scripts/tune_stage3.py --methods fm_cls_guided --datasets dtd
    python scripts/tune_stage3.py --seeds 0 --max-epochs 20   # a quick pass
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.flow_matching.stage3_tuning import (
    build_grid,
    format_tuning_table,
    save_tuning_results,
    search_setting,
    select_best,
)
from src.utils.config import STAGE3_METHODS, Stage3Hyperparams
from src.utils.device import get_device

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The one representative encoder per dataset chosen for Stage 3 (part_3.pdf:
# "choose one representative image encoder for each dataset"). Flowers-102
# has only ResNet-18 in this project; DTD uses DINOv2, the encoder the report
# already treats as representative for it.
STAGE3_SETTINGS = {
    "dtd": "dinov2_vits14",
    "flowers102": "resnet18",
}
K_SHOT = 10
SEEDS = [0, 1, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(STAGE3_METHODS))
    parser.add_argument("--datasets", nargs="+", default=list(STAGE3_SETTINGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--k-shot", type=int, default=K_SHOT)
    parser.add_argument(
        "--max-epochs", type=int, default=None,
        help="Override the epoch budget; useful for a fast smoke pass.",
    )
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "cache"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument(
        "--results-path", default=str(PROJECT_ROOT / "reports" / "stage3_tuning.json")
    )
    args = parser.parse_args()

    device = get_device()
    base = Stage3Hyperparams()
    if args.max_epochs is not None:
        base = Stage3Hyperparams(**{**vars(base), "max_epochs": args.max_epochs})

    total = sum(
        len(build_grid(method)) for method in args.methods
    ) * len(args.datasets) * len(args.seeds)
    print(f"device: {device}   seeds: {args.seeds}   K={args.k_shot}   "
          f"T={base.num_euler_steps}   epochs={base.max_epochs}")
    print(f"{total} runs to train\n")

    completed = {"count": 0}
    started = time.time()

    def progress(run):
        completed["count"] += 1
        elapsed = time.time() - started
        rate = elapsed / completed["count"]
        remaining = (total - completed["count"]) * rate
        print(
            f"  [{completed['count']:>4d}/{total}] {run.method} {run.dataset} "
            f"seed{run.seed} {run.overrides} "
            f"val {run.val_delta * 100:+.2f} test {run.test_delta * 100:+.2f} "
            f"(~{remaining / 60:.0f} min left)",
            flush=True,
        )

    all_summaries = []
    for method in args.methods:
        for dataset in args.datasets:
            encoder = STAGE3_SETTINGS[dataset]
            print(f"== {method} on {dataset}/{encoder} ==", flush=True)

            summaries = search_setting(
                method, dataset, encoder, args.k_shot, args.seeds,
                args.cache_dir, args.output_dir, device,
                base_hyperparams=base, progress=progress,
            )
            all_summaries.extend(summaries)

            best = select_best(summaries)
            print(f"\n{format_tuning_table(summaries)}\n")
            print(f"  selected on validation: {best.label}")
            print(
                f"    val {best.mean_val_delta * 100:+.2f}%   "
                f"test {best.mean_test_delta * 100:+.2f}%   "
                f"accuracy {best.mean_test_accuracy * 100:.2f}%\n",
                flush=True,
            )

    save_tuning_results(all_summaries, args.results_path)
    print(f"Wrote {len(all_summaries)} configurations to {args.results_path}")


if __name__ == "__main__":
    main()
