from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KEY = ["id_student", "code_module", "code_presentation"]
EXPECTED_HORIZONS = (7, 14, 21, 30)
EXPECTED_MODELS = ("Logistic Regression", "Random Forest", "XGBoost")
EXPECTED_ABLATION_GROUPS = (
    "demographics only",
    "engagement only",
    "assessment only",
    "engagement + assessment",
    "full feature set",
)
EXPECTED_RISK_BANDS = ("Low", "Medium", "High", "Critical")


@dataclass
class ResearchValidationSummary:
    processed_dir: Path
    horizon_shapes: pd.DataFrame
    model_pair_count: int
    ablation_row_count: int
    champion_metrics: pd.DataFrame
    risk_band_summary: pd.DataFrame


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_research_outputs(processed_dir: Path) -> ResearchValidationSummary:
    processed_dir = Path(processed_dir)

    required_files = [
        "features_horizon_metadata.csv",
        "model_horizon_comparison.csv",
        "threshold_search_by_horizon.csv",
        "selected_operating_points.csv",
        "champion_test_predictions.csv",
        "champion_test_metrics.csv",
        "ablation_results.csv",
        "ablation_gain_summary.csv",
        "calibration_summary.csv",
        "risk_band_summary.csv",
        "risk_band_test_predictions.csv",
        "model_feature_importance.csv",
        "error_analysis_samples.csv",
    ] + [f"features_prediction_day{h:02d}.csv" for h in EXPECTED_HORIZONS]

    missing_files = [name for name in required_files if not (processed_dir / name).exists()]
    _expect(not missing_files, f"Missing required research outputs: {missing_files}")

    # Horizon prediction tables
    horizon_rows: list[dict[str, int]] = []
    reference_columns: list[str] | None = None
    for horizon in EXPECTED_HORIZONS:
        df = pd.read_csv(processed_dir / f"features_prediction_day{horizon:02d}.csv", low_memory=False)
        horizon_rows.append(
            {
                "horizon_day": horizon,
                "rows": len(df),
                "columns": df.shape[1],
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_keys": int(df.duplicated(subset=KEY).sum()),
            }
        )
        _expect("at_risk" in df.columns, f"`at_risk` missing in day-{horizon} prediction table")
        _expect("horizon_day" in df.columns, f"`horizon_day` missing in day-{horizon} prediction table")
        _expect(df["horizon_day"].nunique() == 1 and int(df["horizon_day"].iloc[0]) == horizon, f"Invalid horizon_day values in day-{horizon} table")
        if reference_columns is None:
            reference_columns = df.columns.tolist()
        else:
            _expect(reference_columns == df.columns.tolist(), f"Schema mismatch detected at day {horizon}")

    horizon_shapes = pd.DataFrame(horizon_rows).sort_values("horizon_day")
    _expect((horizon_shapes["missing_cells"] == 0).all(), "Horizon prediction tables still contain missing values")
    _expect((horizon_shapes["duplicate_keys"] == 0).all(), "Horizon prediction tables contain duplicate enrollment keys")

    # Horizon metadata
    horizon_metadata = pd.read_csv(processed_dir / "features_horizon_metadata.csv", low_memory=False)
    _expect(tuple(sorted(horizon_metadata["horizon_day"].tolist())) == EXPECTED_HORIZONS, "features_horizon_metadata.csv does not cover the expected horizons")
    _expect((horizon_metadata["post_imputation_missing_pct"] == 0).all(), "Post-imputation missingness should be 0 across horizons")

    # Model comparison + selected points
    comparison = pd.read_csv(processed_dir / "model_horizon_comparison.csv", low_memory=False)
    selected = pd.read_csv(processed_dir / "selected_operating_points.csv", low_memory=False)
    threshold_search = pd.read_csv(processed_dir / "threshold_search_by_horizon.csv", low_memory=False)

    expected_pairs = {(h, m) for h in EXPECTED_HORIZONS for m in EXPECTED_MODELS}
    comparison_pairs = set(zip(comparison["horizon_day"], comparison["model"]))
    selected_pairs = set(zip(selected["horizon_day"], selected["model"]))
    threshold_pairs = set(zip(threshold_search["horizon_day"], threshold_search["model"]))

    _expect(comparison_pairs == expected_pairs, "model_horizon_comparison.csv does not contain all horizon-model pairs")
    _expect(selected_pairs == expected_pairs, "selected_operating_points.csv does not contain all horizon-model pairs")
    _expect(threshold_pairs == expected_pairs, "threshold_search_by_horizon.csv does not contain all horizon-model pairs")
    _expect(selected["is_earliest_useful_horizon"].sum() == 1, "There must be exactly one earliest useful horizon flag")

    # Champion metrics
    champion_metrics = pd.read_csv(processed_dir / "champion_test_metrics.csv", low_memory=False)
    _expect(set(champion_metrics["split"]) == {"validation", "test"}, "champion_test_metrics.csv must contain validation and test rows")
    required_metric_cols = {"precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "brier_score"}
    _expect(required_metric_cols.issubset(champion_metrics.columns), "champion_test_metrics.csv is missing required metric columns")

    # Champion predictions
    champion_predictions = pd.read_csv(processed_dir / "champion_test_predictions.csv", low_memory=False)
    _expect(champion_predictions["risk_band"].notna().all(), "champion_test_predictions.csv contains null risk bands")
    _expect(champion_predictions["risk_probability"].between(0, 1).all(), "risk_probability must stay within [0, 1]")
    _expect(champion_predictions["prediction_outcome"].isin({"True Positive", "True Negative", "False Positive", "False Negative"}).all(), "prediction_outcome contains unexpected labels")

    # Ablation
    ablation = pd.read_csv(processed_dir / "ablation_results.csv", low_memory=False)
    ablation_pairs = set(zip(ablation["horizon_day"], ablation["feature_group"]))
    expected_ablation_pairs = {
        (h, group)
        for h in (14, 30)
        for group in EXPECTED_ABLATION_GROUPS
    }
    _expect(ablation_pairs == expected_ablation_pairs, "ablation_results.csv does not contain the expected horizon-feature-group coverage")

    ablation_gain = pd.read_csv(processed_dir / "ablation_gain_summary.csv", low_memory=False)
    _expect(len(ablation_gain) == len(ablation), "ablation_gain_summary.csv row count does not match ablation_results.csv")

    # Calibration and risk bands
    calibration = pd.read_csv(processed_dir / "calibration_summary.csv", low_memory=False)
    _expect(set(calibration["horizon_day"]) == set(EXPECTED_HORIZONS), "calibration_summary.csv must contain all horizons")
    risk_band_summary = pd.read_csv(processed_dir / "risk_band_summary.csv", low_memory=False)
    _expect(tuple(risk_band_summary["risk_band"]) == EXPECTED_RISK_BANDS, "risk_band_summary.csv must contain Low, Medium, High, Critical in order")
    _expect(risk_band_summary["actual_at_risk_rate"].is_monotonic_increasing, "Risk-band actual at-risk rates must be monotonic increasing")
    _expect(risk_band_summary["average_predicted_probability"].is_monotonic_increasing, "Risk-band predicted probabilities must be monotonic increasing")

    risk_band_predictions = pd.read_csv(processed_dir / "risk_band_test_predictions.csv", low_memory=False)
    _expect(len(risk_band_predictions) == len(champion_predictions), "risk_band_test_predictions.csv row count must match champion_test_predictions.csv")

    # Diagnostics
    feature_importance = pd.read_csv(processed_dir / "model_feature_importance.csv", low_memory=False)
    _expect(feature_importance["importance_rank"].min() == 1, "model_feature_importance.csv should start ranking at 1")
    _expect(feature_importance["importance_rank"].is_monotonic_increasing, "importance_rank must be monotonic increasing")

    error_samples = pd.read_csv(processed_dir / "error_analysis_samples.csv", low_memory=False)
    if not error_samples.empty:
        _expect(error_samples["prediction_outcome"].isin({"False Positive", "False Negative"}).all(), "error_analysis_samples.csv must only contain false positives and false negatives")

    return ResearchValidationSummary(
        processed_dir=processed_dir,
        horizon_shapes=horizon_shapes,
        model_pair_count=len(comparison_pairs),
        ablation_row_count=len(ablation),
        champion_metrics=champion_metrics,
        risk_band_summary=risk_band_summary,
    )

