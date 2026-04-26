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
    parser = argparse.ArgumentParser(description="Run the full research pipeline outside notebooks.")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run without writing CSV outputs.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not run the output validator after the pipeline completes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_repo_root()
    sys.path.append(str(root))

    from src.features import build_multi_horizon_feature_store
    from src.models import run_multi_horizon_study
    from src.validation import validate_research_outputs

    processed_dir = root / "data" / "processed"

    print("Step 1/2 - building multi-horizon feature store...")
    build_multi_horizon_feature_store(
        processed_dir=processed_dir,
        write_outputs=not args.no_write,
    )

    print("Step 2/2 - running multi-horizon modeling study...")
    outputs = run_multi_horizon_study(
        processed_dir=processed_dir,
        write_outputs=not args.no_write,
    )

    print()
    print("Champion pair:")
    print(outputs.champion_row[["horizon_day", "model", "threshold", "precision", "recall", "f2", "pr_auc"]].to_string())

    if not args.skip_validation:
        print()
        print("Running acceptance checks...")
        summary = validate_research_outputs(processed_dir)
        print(summary.horizon_shapes.to_string(index=False))
        print()
        print(summary.champion_metrics.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
