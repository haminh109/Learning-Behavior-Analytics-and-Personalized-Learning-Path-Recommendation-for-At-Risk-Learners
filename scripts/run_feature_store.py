from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the multi-horizon feature store.")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[7, 14, 21, 30],
        help="Observation horizons to build.",
    )
    parser.add_argument(
        "--early-window-cap",
        type=int,
        default=14,
        help="Maximum early-engagement window used inside each horizon.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run the feature builder without writing CSV outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_repo_root()
    sys.path.append(str(root))

    from src.features import build_multi_horizon_feature_store

    processed_dir = root / "data" / "processed"
    outputs = build_multi_horizon_feature_store(
        processed_dir=processed_dir,
        horizons=tuple(args.horizons),
        early_window_cap=args.early_window_cap,
        write_outputs=not args.no_write,
    )

    print("Feature store built successfully.")
    print(f"Processed directory: {processed_dir}")
    print()
    print(outputs.features_horizon_metadata.to_string(index=False))

    if not args.no_write:
        print()
        print("Written outputs:")
        for horizon, table in outputs.prediction_tables.items():
            print(f"  features_prediction_day{horizon:02d}.csv -> {table.shape}")
        print(f"  features_final.csv -> {outputs.features_final.shape}")
        print(f"  features_segmentation.csv -> {outputs.features_segmentation.shape}")
        print(f"  features_prediction.csv -> {outputs.features_prediction.shape}")
        print(f"  features_recommendation.csv -> {outputs.features_recommendation.shape}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
