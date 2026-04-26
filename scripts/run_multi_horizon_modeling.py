from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-horizon early-warning modeling study.")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[7, 14, 21, 30],
        help="Prediction horizons to evaluate.",
    )
    parser.add_argument(
        "--ablation-horizons",
        type=int,
        nargs="+",
        default=[14, 30],
        help="Horizons used in the ablation study.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Validation recall floor for threshold selection.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the study without writing CSV outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_repo_root()
    sys.path.append(str(root))

    from src.models import run_multi_horizon_study

    processed_dir = root / "data" / "processed"
    outputs = run_multi_horizon_study(
        processed_dir=processed_dir,
        horizons=tuple(args.horizons),
        ablation_horizons=tuple(args.ablation_horizons),
        target_recall=args.target_recall,
        write_outputs=not args.no_write,
    )

    champion_metrics = outputs.champion_test_metrics[
        ["split", "horizon_day", "model", "threshold", "precision", "recall", "f2", "roc_auc", "pr_auc", "brier_score"]
    ]

    print("Multi-horizon modeling study completed.")
    print()
    print("Champion pair:")
    print(outputs.champion_row[["horizon_day", "model", "threshold", "precision", "recall", "f2", "pr_auc"]].to_string())
    print()
    print(champion_metrics.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
