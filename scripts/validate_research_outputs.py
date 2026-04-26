from __future__ import annotations

import sys
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "data").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate repository root.")


def main() -> int:
    root = find_repo_root()
    sys.path.append(str(root))

    from src.validation import validate_research_outputs

    processed_dir = root / "data" / "processed"
    summary = validate_research_outputs(processed_dir)

    print("Research output validation passed.")
    print()
    print(summary.horizon_shapes.to_string(index=False))
    print()
    print(summary.champion_metrics.to_string(index=False))
    print()
    print(summary.risk_band_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
