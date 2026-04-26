from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

KEY = ["id_student", "code_module", "code_presentation"]
TARGET_COL = "at_risk"
OUTCOME_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
RISK_BAND_LABELS = ["Low", "Medium", "High", "Critical"]


@dataclass
class ModelingOutputs:
    model_horizon_comparison: pd.DataFrame
    threshold_search_by_horizon: pd.DataFrame
    selected_operating_points: pd.DataFrame
    champion_test_predictions: pd.DataFrame
    champion_test_metrics: pd.DataFrame
    ablation_results: pd.DataFrame
    ablation_gain_summary: pd.DataFrame
    calibration_summary: pd.DataFrame
    risk_band_summary: pd.DataFrame
    risk_band_test_predictions: pd.DataFrame
    model_feature_importance: pd.DataFrame
    error_analysis_samples: pd.DataFrame
    segment_model_performance: pd.DataFrame
    outcome_risk_summary: pd.DataFrame
    champion_row: pd.Series
    split_summary: pd.DataFrame


def _fbeta_from_precision_recall(precision: float, recall: float, beta: float = 2.0) -> float:
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)


def _evaluate_threshold(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = _fbeta_from_precision_recall(precision, recall, beta=2.0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _choose_operating_point(model_rows: list[dict[str, float]], target_recall: float) -> dict[str, float]:
    frame = pd.DataFrame(model_rows).copy()
    frame["meets_target_recall"] = frame["recall"] >= target_recall
    pool = frame.loc[frame["meets_target_recall"]].copy()
    if pool.empty:
        pool = frame.copy()
    selected = pool.sort_values(
        ["precision", "f2", "pr_auc", "threshold"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return selected.to_dict()


def _sanitize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    sanitize_map = {col: re.sub(r"[^0-9a-zA-Z_]+", "_", col) for col in df.columns}
    reverse_map = {sanitized: original for original, sanitized in sanitize_map.items()}
    return df.rename(columns=sanitize_map), reverse_map


def _build_split_map(processed_dir: Path, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    student_info = pd.read_csv(processed_dir / "student_info_clean.csv", low_memory=False)
    split_source = student_info[KEY + ["final_result"]].copy()
    split_source[TARGET_COL] = split_source["final_result"].isin(["Fail", "Withdrawn"]).astype(int)

    train_keys, test_keys = train_test_split(
        split_source,
        test_size=0.20,
        stratify=split_source[TARGET_COL],
        random_state=random_state,
    )
    train_keys, val_keys = train_test_split(
        train_keys,
        test_size=0.25,
        stratify=train_keys[TARGET_COL],
        random_state=random_state,
    )

    split_map = pd.concat(
        [
            train_keys[KEY].assign(split="train"),
            val_keys[KEY].assign(split="validation"),
            test_keys[KEY].assign(split="test"),
        ],
        ignore_index=True,
    )

    split_summary = (
        split_source.merge(split_map, on=KEY, how="left")
        .groupby("split", as_index=False)
        .agg(rows=(TARGET_COL, "size"), at_risk_rate=(TARGET_COL, "mean"))
        .sort_values("split")
    )
    return split_map, split_summary


def _load_prediction_tables(processed_dir: Path, horizons: tuple[int, ...]) -> dict[int, pd.DataFrame]:
    return {
        horizon: pd.read_csv(
            processed_dir / f"features_prediction_day{horizon:02d}.csv",
            low_memory=False,
        )
        for horizon in horizons
    }


def _make_candidate_models(scale_pos_weight: float, random_state: int) -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.80,
            colsample_bytree=0.80,
            scale_pos_weight=max(scale_pos_weight, 1.0),
            min_child_weight=2,
            reg_lambda=1.5,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=4,
        ),
    }


def _select_champion(
    selected_points: pd.DataFrame,
    target_recall: float,
    pr_auc_tolerance: float = 0.01,
    f2_tolerance: float = 0.01,
) -> pd.Series:
    candidates = selected_points.loc[selected_points["recall"] >= target_recall].copy()
    if candidates.empty:
        candidates = selected_points.copy()

    best_pr_auc = float(candidates["pr_auc"].max())
    near_best = candidates.loc[candidates["pr_auc"] >= best_pr_auc - pr_auc_tolerance].copy()
    best_f2 = float(near_best["f2"].max())
    near_best = near_best.loc[near_best["f2"] >= best_f2 - f2_tolerance].copy()

    champion = near_best.sort_values(
        ["horizon_day", "pr_auc", "f2", "precision", "recall"],
        ascending=[True, False, False, False, False],
    ).iloc[0]
    return champion


def _build_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    demographic_cols = [
        col
        for col in columns
        if col in {"num_of_prev_attempts", "studied_credits"}
        or col.startswith("gender_")
        or col.startswith("highest_education_")
        or col.startswith("imd_band_")
        or col.startswith("age_band_")
        or col.startswith("disability_")
    ]
    engagement_cols = [
        col
        for col in columns
        if col
        in {
            "total_clicks_log",
            "active_days_log",
            "early_engagement_log",
            "early_engagement_ratio",
            "days_since_last",
            "engagement_intensity_log",
            "persistence_score",
        }
    ]
    assessment_cols = [
        col
        for col in columns
        if col
        in {
            "avg_score",
            "score_std",
            "avg_submission_delay",
            "has_submission_by_horizon",
            "assessment_discipline",
            "num_submitted",
            "completion_ratio",
        }
    ]

    groups = {
        "demographics only": demographic_cols,
        "engagement only": engagement_cols,
        "assessment only": assessment_cols,
        "engagement + assessment": engagement_cols + assessment_cols,
        "full feature set": columns,
    }
    return groups


def _prepare_band_edges(validation_prob: np.ndarray, min_share: float = 0.05) -> tuple[list[float], str]:
    fixed_edges = [-np.inf, 0.25, 0.50, 0.75, np.inf]
    fixed_bands = pd.Series(
        pd.cut(
            validation_prob,
            bins=fixed_edges,
            labels=RISK_BAND_LABELS,
            right=False,
            include_lowest=True,
        )
    )
    fixed_shares = fixed_bands.value_counts(sort=False) / max(len(fixed_bands), 1)
    if not fixed_shares.empty and fixed_shares.min() >= min_share:
        return fixed_edges, "fixed_cutpoints"

    quantile_values = np.quantile(validation_prob, [0.0, 0.25, 0.50, 0.75, 1.0])
    quantile_values[0] = -np.inf
    quantile_values[-1] = np.inf
    if len(np.unique(quantile_values)) == 5:
        return quantile_values.tolist(), "validation_quantiles"
    return fixed_edges, "fixed_cutpoints_fallback"


def _assign_risk_bands(probabilities: np.ndarray, edges: list[float]) -> pd.Categorical:
    return pd.cut(
        probabilities,
        bins=edges,
        labels=RISK_BAND_LABELS,
        right=False,
        include_lowest=True,
    )


def _probability_bin_summary(
    y_true: pd.Series,
    y_prob: np.ndarray,
    horizon_day: int,
    model_name: str,
    split_label: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true, "y_prob": y_prob}).copy()
    frame["probability_bin"] = pd.cut(
        frame["y_prob"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
        duplicates="drop",
    )
    summary = (
        frame.groupby("probability_bin", observed=False)
        .agg(
            n=("y_true", "size"),
            avg_predicted_probability=("y_prob", "mean"),
            observed_at_risk_rate=("y_true", "mean"),
        )
        .reset_index()
    )
    summary["horizon_day"] = horizon_day
    summary["model"] = model_name
    summary["split"] = split_label
    summary["brier_score"] = brier_score_loss(y_true, y_prob)
    return summary[
        [
            "horizon_day",
            "model",
            "split",
            "probability_bin",
            "n",
            "avg_predicted_probability",
            "observed_at_risk_rate",
            "brier_score",
        ]
    ]


def _native_feature_importance(model: object, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_)).ravel()
    else:
        importance = np.full(len(feature_names), np.nan, dtype=float)

    return pd.DataFrame({"feature": feature_names, "native_importance": importance})


def run_multi_horizon_study(
    processed_dir: Path,
    horizons: tuple[int, ...] = (7, 14, 21, 30),
    ablation_horizons: tuple[int, ...] = (14, 30),
    random_state: int = 42,
    target_recall: float = 0.90,
    threshold_grid: np.ndarray | None = None,
    write_outputs: bool = True,
) -> ModelingOutputs:
    processed_dir = Path(processed_dir)
    threshold_grid = threshold_grid if threshold_grid is not None else np.arange(0.05, 0.81, 0.05)

    split_map, split_summary = _build_split_map(processed_dir, random_state=random_state)
    prediction_tables = _load_prediction_tables(processed_dir, horizons=horizons)
    segment_assignments = pd.read_csv(processed_dir / "segment_assignments.csv", low_memory=False)
    learning_paths = pd.read_csv(processed_dir / "personalized_learning_paths.csv", low_memory=False)

    scale_pos_weight = (
        split_map.merge(
            pd.read_csv(processed_dir / "student_info_clean.csv", low_memory=False)[KEY + ["final_result"]],
            on=KEY,
            how="left",
        )
        .assign(at_risk=lambda df: df["final_result"].isin(["Fail", "Withdrawn"]).astype(int))
        .query("split == 'train'")["at_risk"]
    )
    scale_pos_weight = float((scale_pos_weight.eq(0).sum()) / max(scale_pos_weight.eq(1).sum(), 1))

    threshold_rows: list[dict[str, float]] = []
    selected_rows: list[dict[str, float]] = []
    comparison_rows: list[dict[str, float]] = []
    artifacts: dict[tuple[int, str], dict[str, object]] = {}

    for horizon, raw_prediction in prediction_tables.items():
        modeling = raw_prediction.merge(split_map, on=KEY, how="left")
        meta_cols = KEY + ["split", "horizon_day"]
        feature_frame = modeling.drop(columns=meta_cols + [TARGET_COL])
        feature_frame, reverse_map = _sanitize_columns(feature_frame)
        feature_frame = feature_frame.astype("float32")
        target = modeling[TARGET_COL].copy()

        split_mask = modeling["split"]
        X_train = feature_frame.loc[split_mask.eq("train")].copy()
        X_val = feature_frame.loc[split_mask.eq("validation")].copy()
        X_test = feature_frame.loc[split_mask.eq("test")].copy()
        y_train = target.loc[split_mask.eq("train")].copy()
        y_val = target.loc[split_mask.eq("validation")].copy()
        y_test = target.loc[split_mask.eq("test")].copy()
        meta_test = raw_prediction.loc[split_mask.eq("test"), KEY + ["horizon_day"]].reset_index(drop=True)

        candidate_models = _make_candidate_models(scale_pos_weight=scale_pos_weight, random_state=random_state)

        for model_name, model in candidate_models.items():
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                fitted = model.fit(X_train, y_train)
                val_prob = fitted.predict_proba(X_val)[:, 1]
                test_prob = fitted.predict_proba(X_test)[:, 1]

            model_search_rows: list[dict[str, float]] = []
            for threshold in threshold_grid:
                row = _evaluate_threshold(y_val, val_prob, float(threshold))
                row.update({"horizon_day": horizon, "model": model_name})
                model_search_rows.append(row)
                threshold_rows.append(row)

            selected = _choose_operating_point(model_search_rows, target_recall=target_recall)
            selected.update(
                {
                    "horizon_day": horizon,
                    "model": model_name,
                    "feature_count": int(X_train.shape[1]),
                }
            )
            selected_rows.append(selected)

            test_eval = _evaluate_threshold(y_test, test_prob, float(selected["threshold"]))
            comparison_rows.append(
                {
                    "horizon_day": horizon,
                    "model": model_name,
                    "selected_threshold": float(selected["threshold"]),
                    "validation_accuracy": float(selected["accuracy"]),
                    "validation_precision": float(selected["precision"]),
                    "validation_recall": float(selected["recall"]),
                    "validation_f1": float(selected["f1"]),
                    "validation_f2": float(selected["f2"]),
                    "validation_roc_auc": float(selected["roc_auc"]),
                    "validation_pr_auc": float(selected["pr_auc"]),
                    "validation_brier_score": float(selected["brier_score"]),
                    "meets_target_recall": bool(selected["recall"] >= target_recall),
                    "test_accuracy": float(test_eval["accuracy"]),
                    "test_precision": float(test_eval["precision"]),
                    "test_recall": float(test_eval["recall"]),
                    "test_f1": float(test_eval["f1"]),
                    "test_f2": float(test_eval["f2"]),
                    "test_roc_auc": float(test_eval["roc_auc"]),
                    "test_pr_auc": float(test_eval["pr_auc"]),
                    "test_brier_score": float(test_eval["brier_score"]),
                    "test_tn": int(test_eval["tn"]),
                    "test_fp": int(test_eval["fp"]),
                    "test_fn": int(test_eval["fn"]),
                    "test_tp": int(test_eval["tp"]),
                }
            )

            artifacts[(horizon, model_name)] = {
                "model": fitted,
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "val_prob": val_prob,
                "test_prob": test_prob,
                "meta_test": meta_test,
                "raw_test_features": raw_prediction.loc[split_mask.eq("test")].reset_index(drop=True),
                "reverse_feature_map": reverse_map,
            }

    threshold_search = pd.DataFrame(threshold_rows).sort_values(["horizon_day", "model", "threshold"])
    selected_operating_points = pd.DataFrame(selected_rows).sort_values(
        ["horizon_day", "model"]
    )
    selected_operating_points["is_best_for_horizon"] = False
    for horizon in horizons:
        horizon_rows = selected_operating_points.loc[selected_operating_points["horizon_day"].eq(horizon)]
        if horizon_rows.empty:
            continue
        best_idx = horizon_rows.sort_values(
            ["recall", "pr_auc", "f2", "precision"],
            ascending=[False, False, False, False],
        ).index[0]
        selected_operating_points.loc[best_idx, "is_best_for_horizon"] = True

    eligible_best = selected_operating_points.loc[
        selected_operating_points["is_best_for_horizon"] & selected_operating_points["recall"].ge(target_recall)
    ]
    earliest_useful_horizon = int(eligible_best["horizon_day"].min()) if not eligible_best.empty else int(min(horizons))
    selected_operating_points["is_earliest_useful_horizon"] = (
        selected_operating_points["is_best_for_horizon"] & selected_operating_points["horizon_day"].eq(earliest_useful_horizon)
    )

    model_horizon_comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_pr_auc", "validation_f2", "horizon_day"],
        ascending=[False, False, True],
    )

    champion_row = _select_champion(selected_operating_points, target_recall=target_recall)
    champion_horizon = int(champion_row["horizon_day"])
    champion_model_name = str(champion_row["model"])
    champion_artifact = artifacts[(champion_horizon, champion_model_name)]

    champion_val_metrics = _evaluate_threshold(
        champion_artifact["y_val"],
        champion_artifact["val_prob"],
        float(champion_row["threshold"]),
    )
    champion_test_metrics_dict = _evaluate_threshold(
        champion_artifact["y_test"],
        champion_artifact["test_prob"],
        float(champion_row["threshold"]),
    )
    champion_test_pred = (champion_artifact["test_prob"] >= float(champion_row["threshold"])).astype(int)

    champion_test_metrics = pd.DataFrame(
        [
            {
                "split": "validation",
                "horizon_day": champion_horizon,
                "model": champion_model_name,
                **champion_val_metrics,
            },
            {
                "split": "test",
                "horizon_day": champion_horizon,
                "model": champion_model_name,
                **champion_test_metrics_dict,
            },
        ]
    )

    band_edges, band_method = _prepare_band_edges(champion_artifact["val_prob"])
    risk_bands = _assign_risk_bands(champion_artifact["test_prob"], band_edges)

    champion_test_predictions = champion_artifact["meta_test"].copy()
    champion_test_predictions["model"] = champion_model_name
    champion_test_predictions["selected_threshold"] = float(champion_row["threshold"])
    champion_test_predictions["y_true"] = champion_artifact["y_test"].values
    champion_test_predictions["risk_probability"] = champion_artifact["test_prob"]
    champion_test_predictions["y_pred"] = champion_test_pred
    champion_test_predictions["risk_band"] = risk_bands.astype(str)
    champion_test_predictions["risk_band_method"] = band_method
    champion_test_predictions = champion_test_predictions.merge(
        segment_assignments[KEY + ["final_result", "cluster_label", "rule_segment"]],
        on=KEY,
        how="left",
    )
    champion_test_predictions = champion_test_predictions.merge(
        learning_paths[
            KEY
            + [
                "recommended_path",
                "action_1",
                "action_2",
                "action_3",
                "recommendation_score",
            ]
        ],
        on=KEY,
        how="left",
    )
    champion_test_predictions["prediction_outcome"] = np.select(
        [
            champion_test_predictions["y_true"].eq(1) & champion_test_predictions["y_pred"].eq(1),
            champion_test_predictions["y_true"].eq(0) & champion_test_predictions["y_pred"].eq(0),
            champion_test_predictions["y_true"].eq(0) & champion_test_predictions["y_pred"].eq(1),
            champion_test_predictions["y_true"].eq(1) & champion_test_predictions["y_pred"].eq(0),
        ],
        ["True Positive", "True Negative", "False Positive", "False Negative"],
        default="Unknown",
    )

    risk_band_test_predictions = champion_test_predictions.copy()
    risk_band_summary = (
        risk_band_test_predictions.groupby("risk_band", observed=False)
        .agg(
            n=("id_student", "size"),
            actual_at_risk_rate=("y_true", "mean"),
            average_predicted_probability=("risk_probability", "mean"),
        )
        .reindex(RISK_BAND_LABELS)
        .reset_index()
    )
    risk_band_summary["risk_band_method"] = band_method
    risk_band_summary["band_edges"] = str(band_edges)

    calibration_frames: list[pd.DataFrame] = []
    for horizon in horizons:
        if (horizon, champion_model_name) not in artifacts:
            continue
        artifact = artifacts[(horizon, champion_model_name)]
        calibration_frames.append(
            _probability_bin_summary(
                y_true=artifact["y_test"],
                y_prob=artifact["test_prob"],
                horizon_day=horizon,
                model_name=champion_model_name,
                split_label="test",
            )
        )

    calibration_summary = pd.concat(calibration_frames, ignore_index=True)

    native_importance = _native_feature_importance(
        champion_artifact["model"],
        feature_names=champion_artifact["X_val"].columns.tolist(),
    )
    perm = permutation_importance(
        champion_artifact["model"],
        champion_artifact["X_val"],
        champion_artifact["y_val"],
        n_repeats=5,
        random_state=random_state,
        scoring="average_precision",
        n_jobs=1,
    )
    permutation_df = pd.DataFrame(
        {
            "feature": champion_artifact["X_val"].columns,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )
    model_feature_importance = native_importance.merge(permutation_df, on="feature", how="left")
    model_feature_importance["original_feature_name"] = model_feature_importance["feature"].map(
        champion_artifact["reverse_feature_map"]
    )
    model_feature_importance["horizon_day"] = champion_horizon
    model_feature_importance["model"] = champion_model_name
    model_feature_importance = model_feature_importance.sort_values(
        ["permutation_importance_mean", "native_importance"],
        ascending=[False, False],
    ).reset_index(drop=True)
    model_feature_importance["importance_rank"] = np.arange(1, len(model_feature_importance) + 1)

    error_analysis = champion_test_predictions.loc[
        champion_test_predictions["prediction_outcome"].isin(["False Positive", "False Negative"])
    ].copy()
    error_analysis = error_analysis.merge(
        champion_artifact["raw_test_features"][
            KEY
            + [
                "days_since_last",
                "avg_score",
                "avg_submission_delay",
                "completion_ratio",
                "num_submitted",
                "total_clicks_log",
                "active_days_log",
                "early_engagement_ratio",
                "assessment_discipline",
                "persistence_score",
                "learning_risk_index",
            ]
        ],
        on=KEY,
        how="left",
    )
    error_analysis["distance_to_threshold"] = (
        error_analysis["risk_probability"] - float(champion_row["threshold"])
    )
    false_negative = error_analysis.loc[error_analysis["prediction_outcome"].eq("False Negative")].sort_values(
        ["risk_probability", "recommendation_score"],
        ascending=[False, False],
    )
    false_positive = error_analysis.loc[error_analysis["prediction_outcome"].eq("False Positive")].sort_values(
        ["risk_probability", "recommendation_score"],
        ascending=[False, False],
    )
    error_analysis_samples = pd.concat([false_negative.head(20), false_positive.head(20)], ignore_index=True)

    feature_groups = _build_feature_groups(champion_artifact["X_train"].columns.tolist())
    ablation_rows: list[dict[str, float]] = []
    for horizon in ablation_horizons:
        full_artifact = artifacts[(horizon, champion_model_name)]
        for group_name, selected_features in feature_groups.items():
            candidate_model = _make_candidate_models(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            )[champion_model_name]
            X_train = full_artifact["X_train"][selected_features].copy()
            X_val = full_artifact["X_val"][selected_features].copy()
            X_test = full_artifact["X_test"][selected_features].copy()

            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                fitted = candidate_model.fit(X_train, full_artifact["y_train"])
                val_prob = fitted.predict_proba(X_val)[:, 1]
                test_prob = fitted.predict_proba(X_test)[:, 1]

            group_search_rows: list[dict[str, float]] = []
            for threshold in threshold_grid:
                group_search_rows.append(_evaluate_threshold(full_artifact["y_val"], val_prob, float(threshold)))
            selected = _choose_operating_point(group_search_rows, target_recall=target_recall)
            test_eval = _evaluate_threshold(full_artifact["y_test"], test_prob, float(selected["threshold"]))
            ablation_rows.append(
                {
                    "horizon_day": horizon,
                    "model": champion_model_name,
                    "feature_group": group_name,
                    "feature_count": len(selected_features),
                    "selected_threshold": float(selected["threshold"]),
                    "validation_precision": float(selected["precision"]),
                    "validation_recall": float(selected["recall"]),
                    "validation_f2": float(selected["f2"]),
                    "validation_roc_auc": float(selected["roc_auc"]),
                    "validation_pr_auc": float(selected["pr_auc"]),
                    "test_precision": float(test_eval["precision"]),
                    "test_recall": float(test_eval["recall"]),
                    "test_f2": float(test_eval["f2"]),
                    "test_roc_auc": float(test_eval["roc_auc"]),
                    "test_pr_auc": float(test_eval["pr_auc"]),
                }
            )

    ablation_results = pd.DataFrame(ablation_rows).sort_values(["horizon_day", "feature_group"])
    ablation_gain_summary = ablation_results.copy()
    full_baseline = (
        ablation_results.loc[ablation_results["feature_group"].eq("full feature set")]
        .set_index("horizon_day")
        .add_prefix("full_")
    )
    ablation_gain_summary = ablation_gain_summary.merge(
        full_baseline,
        left_on="horizon_day",
        right_index=True,
        how="left",
    )
    for metric in ["test_precision", "test_recall", "test_f2", "test_roc_auc", "test_pr_auc"]:
        ablation_gain_summary[f"delta_{metric}_vs_full"] = (
            ablation_gain_summary[metric] - ablation_gain_summary[f"full_{metric}"]
        )

    segment_model_performance = (
        champion_test_predictions.groupby("cluster_label", observed=True)[
            ["y_true", "y_pred", "risk_probability"]
        ]
        .apply(
            lambda g: pd.Series(
                {
                    "n": len(g),
                    "actual_at_risk_rate": g["y_true"].mean(),
                    "predicted_positive_rate": g["y_pred"].mean(),
                    "average_risk_probability": g["risk_probability"].mean(),
                    "precision": precision_score(g["y_true"], g["y_pred"], zero_division=0),
                    "recall": recall_score(g["y_true"], g["y_pred"], zero_division=0),
                    "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                }
            )
        )
        .reset_index()
        .sort_values("actual_at_risk_rate", ascending=False)
    )

    outcome_risk_summary = (
        champion_test_predictions.groupby("final_result", observed=True)
        .agg(
            n=("id_student", "size"),
            predicted_positive_rate=("y_pred", "mean"),
            average_risk_probability=("risk_probability", "mean"),
        )
        .reindex(OUTCOME_ORDER)
        .reset_index()
    )

    if write_outputs:
        model_horizon_comparison.to_csv(processed_dir / "model_horizon_comparison.csv", index=False)
        threshold_search.to_csv(processed_dir / "threshold_search_by_horizon.csv", index=False)
        selected_operating_points.to_csv(processed_dir / "selected_operating_points.csv", index=False)
        champion_test_predictions.to_csv(processed_dir / "champion_test_predictions.csv", index=False)
        champion_test_metrics.to_csv(processed_dir / "champion_test_metrics.csv", index=False)
        ablation_results.to_csv(processed_dir / "ablation_results.csv", index=False)
        ablation_gain_summary.to_csv(processed_dir / "ablation_gain_summary.csv", index=False)
        calibration_summary.to_csv(processed_dir / "calibration_summary.csv", index=False)
        risk_band_summary.to_csv(processed_dir / "risk_band_summary.csv", index=False)
        risk_band_test_predictions.to_csv(processed_dir / "risk_band_test_predictions.csv", index=False)
        model_feature_importance.to_csv(processed_dir / "model_feature_importance.csv", index=False)
        error_analysis_samples.to_csv(processed_dir / "error_analysis_samples.csv", index=False)
        segment_model_performance.to_csv(processed_dir / "segment_model_performance.csv", index=False)
        outcome_risk_summary.to_csv(processed_dir / "outcome_risk_summary.csv", index=False)

    return ModelingOutputs(
        model_horizon_comparison=model_horizon_comparison,
        threshold_search_by_horizon=threshold_search,
        selected_operating_points=selected_operating_points,
        champion_test_predictions=champion_test_predictions,
        champion_test_metrics=champion_test_metrics,
        ablation_results=ablation_results,
        ablation_gain_summary=ablation_gain_summary,
        calibration_summary=calibration_summary,
        risk_band_summary=risk_band_summary,
        risk_band_test_predictions=risk_band_test_predictions,
        model_feature_importance=model_feature_importance,
        error_analysis_samples=error_analysis_samples,
        segment_model_performance=segment_model_performance,
        outcome_risk_summary=outcome_risk_summary,
        champion_row=champion_row,
        split_summary=split_summary,
    )
