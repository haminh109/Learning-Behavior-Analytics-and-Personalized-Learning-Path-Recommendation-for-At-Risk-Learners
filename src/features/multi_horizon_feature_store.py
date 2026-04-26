from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

KEY = ["id_student", "code_module", "code_presentation"]
AT_RISK_OUTCOMES = {"Fail", "Withdrawn"}
CAT_ENCODE_COLS = ["gender", "highest_education", "imd_band", "age_band", "disability"]
SEG_FEATURES = [
    "total_clicks_log",
    "active_days_log",
    "early_engagement_log",
    "early_engagement_ratio",
    "avg_score",
    "score_std",
    "engagement_intensity_log",
    "assessment_discipline",
    "persistence_score",
    "completion_ratio",
]
PRED_NUM_FEATURES = [
    "total_clicks_log",
    "active_days_log",
    "early_engagement_log",
    "early_engagement_ratio",
    "days_since_last",
    "avg_score",
    "score_std",
    "avg_submission_delay",
    "has_submission_by_horizon",
    "engagement_intensity_log",
    "assessment_discipline",
    "persistence_score",
    "learning_risk_index",
    "num_of_prev_attempts",
    "studied_credits",
    "num_submitted",
    "completion_ratio",
]
REC_FEATURES = [
    "total_clicks_log",
    "early_engagement_ratio",
    "engagement_intensity_log",
    "avg_score",
    "completion_ratio",
    "assessment_discipline",
    "persistence_score",
    "learning_risk_index",
]
FULL_FEATURE_COLUMNS = KEY + [
    "horizon_day",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "num_of_prev_attempts",
    "studied_credits",
    "disability",
    "final_result",
    "at_risk",
    "total_clicks",
    "active_days",
    "last_activity_date",
    "days_since_last",
    "first_activity_date",
    "activity_span",
    "early_engagement",
    "early_engagement_ratio",
    "has_submission_by_day30",
    "avg_submission_delay",
    "avg_score",
    "score_std",
    "num_submitted",
    "num_total_assessments",
    "completion_ratio",
    "delay_norm",
    "total_clicks_log",
    "active_days_log",
    "early_engagement_log",
    "engagement_intensity_log",
    "engagement_intensity",
    "assessment_discipline",
    "persistence_score",
    "learning_risk_index",
]


@dataclass
class FeatureStoreOutputs:
    features_by_horizon: dict[int, pd.DataFrame]
    prediction_tables: dict[int, pd.DataFrame]
    features_horizon_metadata: pd.DataFrame
    features_final: pd.DataFrame
    features_segmentation: pd.DataFrame
    features_prediction: pd.DataFrame
    features_recommendation: pd.DataFrame


def _minmax_norm(series: pd.Series) -> pd.Series:
    series = series.fillna(0)
    mn = float(series.min())
    mx = float(series.max())
    if mx == mn:
        return pd.Series(0.5, index=series.index, dtype=float)
    return (series - mn) / (mx - mn)


def _weighted_score_agg(group: pd.DataFrame) -> float:
    weights = group["weight_effective"].fillna(0)
    scores = group["score"].astype(float)
    weight_sum = float(weights.sum())
    if weight_sum > 0:
        return float(np.average(scores, weights=weights))
    return float(scores.mean())


def _load_core_tables(processed_dir: Path) -> dict[str, pd.DataFrame]:
    student_info = pd.read_csv(processed_dir / "student_info_clean.csv", low_memory=False)
    assessment_performance = pd.read_csv(processed_dir / "assessment_performance.csv", low_memory=False)
    assessments_clean = pd.read_csv(processed_dir / "assessments_clean.csv", low_memory=False)
    student_vle = pd.read_csv(
        processed_dir / "student_vle_clean.csv",
        low_memory=False,
        usecols=KEY + ["date", "sum_click"],
        dtype={
            "id_student": "int32",
            "date": "int16",
            "sum_click": "int32",
        },
    )
    return {
        "student_info": student_info,
        "assessment_performance": assessment_performance,
        "assessments_clean": assessments_clean,
        "student_vle": student_vle,
    }


def _build_categorical_template(student_info: pd.DataFrame) -> pd.DataFrame:
    base = student_info[KEY + CAT_ENCODE_COLS].copy()
    base[CAT_ENCODE_COLS] = base[CAT_ENCODE_COLS].fillna("Unknown")
    encoded = pd.get_dummies(base[CAT_ENCODE_COLS], drop_first=True, dtype=int)
    return pd.concat([base[KEY].reset_index(drop=True), encoded.reset_index(drop=True)], axis=1)


def _build_vle_features_by_horizon(
    student_vle: pd.DataFrame,
    horizons: tuple[int, ...],
    early_window_cap: int,
) -> dict[int, pd.DataFrame]:
    max_horizon = max(horizons)
    daily_vle = (
        student_vle.loc[student_vle["date"].between(0, max_horizon), KEY + ["date", "sum_click"]]
        .groupby(KEY + ["date"], as_index=False, observed=True)["sum_click"]
        .sum()
    )

    results: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        scope = daily_vle.loc[daily_vle["date"] <= horizon].copy()
        agg = scope.groupby(KEY, as_index=False, observed=True).agg(
            total_clicks=("sum_click", "sum"),
            active_days=("date", "size"),
            first_activity_date=("date", "min"),
            last_activity_date=("date", "max"),
        )

        early_cutoff = min(horizon, early_window_cap)
        early = (
            scope.loc[scope["date"] <= early_cutoff]
            .groupby(KEY, as_index=False, observed=True)["sum_click"]
            .sum()
            .rename(columns={"sum_click": "early_engagement"})
        )

        frame = agg.merge(early, on=KEY, how="left")
        frame["days_since_last"] = np.where(
            frame["last_activity_date"].notna(),
            horizon - frame["last_activity_date"],
            horizon + 1,
        )
        results[horizon] = frame

    return results


def _prepare_assessment_table(
    assessment_performance: pd.DataFrame,
    assessments_clean: pd.DataFrame,
) -> pd.DataFrame:
    assessment_meta = assessments_clean[["id_assessment", "date", "weight"]].rename(
        columns={"date": "deadline", "weight": "weight_meta"}
    )
    assess = assessment_performance.merge(assessment_meta, on="id_assessment", how="left")
    assess = assess.loc[
        assess["is_banked"].eq(0)
        & assess["deadline"].notna()
        & assess["date_submitted"].notna()
    ].copy()

    assess["deadline"] = assess["deadline"].astype(float)
    assess["date_submitted"] = assess["date_submitted"].astype(float)
    assess["weight_effective"] = assess["weight"].fillna(assess["weight_meta"]).fillna(0)
    return assess


def _build_assessment_features_by_horizon(
    assess: pd.DataFrame,
    assessments_clean: pd.DataFrame,
    horizons: tuple[int, ...],
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    results: dict[int, pd.DataFrame] = {}
    total_assessments: dict[int, pd.DataFrame] = {}

    for horizon in horizons:
        scope = assess.loc[assess["date_submitted"] <= horizon].copy()
        scope["submission_delay"] = scope["deadline"] - scope["date_submitted"]

        submitted = (
            scope.groupby(KEY, as_index=False)["id_assessment"]
            .count()
            .rename(columns={"id_assessment": "num_submitted"})
        )

        avg_delay = (
            scope.groupby(KEY, as_index=False)["submission_delay"]
            .mean()
            .rename(columns={"submission_delay": "avg_submission_delay"})
        )

        scored = scope.loc[scope["score"].notna()].copy()
        weighted_score = (
            scored.groupby(KEY)[["score", "weight_effective"]]
            .apply(_weighted_score_agg)
            .reset_index(name="avg_score")
        )
        score_std = (
            scored.groupby(KEY, as_index=False)["score"]
            .std()
            .rename(columns={"score": "score_std"})
        )
        score_std["score_std"] = score_std["score_std"].fillna(0)

        assess_features = submitted.merge(avg_delay, on=KEY, how="outer")
        assess_features = assess_features.merge(weighted_score, on=KEY, how="outer")
        assess_features = assess_features.merge(score_std, on=KEY, how="outer")
        results[horizon] = assess_features

        total_assess = (
            assessments_clean.loc[assessments_clean["date"].fillna(np.inf) <= horizon]
            .groupby(["code_module", "code_presentation"], as_index=False)["id_assessment"]
            .count()
            .rename(columns={"id_assessment": "num_total_assessments"})
        )
        total_assessments[horizon] = total_assess

    return results, total_assessments


def _build_features_for_horizon(
    student_info: pd.DataFrame,
    vle_features: pd.DataFrame,
    assessment_features: pd.DataFrame,
    total_assessments: pd.DataFrame,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    features = student_info[
        KEY
        + [
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "final_result",
        ]
    ].copy()

    features = features.merge(vle_features, on=KEY, how="left")
    features = features.merge(assessment_features, on=KEY, how="left")
    features = features.merge(total_assessments, on=["code_module", "code_presentation"], how="left")

    raw_missing_pct = float(features.isna().mean().mean() * 100)
    activity_coverage = float(features["total_clicks"].notna().mean())
    score_coverage = float(features["avg_score"].notna().mean())

    features["horizon_day"] = horizon
    features["early_engagement"] = features["early_engagement"].fillna(0)
    features["early_engagement_ratio"] = (
        features["early_engagement"] / features["total_clicks"].replace(0, np.nan)
    ).clip(upper=1).fillna(0)

    features["engagement_intensity"] = (
        features["total_clicks"] / features["active_days"].replace(0, np.nan)
    ).fillna(0)

    features["num_submitted"] = features["num_submitted"].fillna(0)
    features["num_total_assessments"] = features["num_total_assessments"].fillna(0)
    features["has_submission_by_horizon"] = features["num_submitted"].gt(0).astype(int)
    features["completion_ratio"] = (
        features["num_submitted"] / features["num_total_assessments"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    delay_non_null = features.loc[
        features["has_submission_by_horizon"].eq(1), "avg_submission_delay"
    ].dropna()
    delay_impute_value = float(delay_non_null.median()) if not delay_non_null.empty else 0.0
    features["avg_submission_delay"] = features["avg_submission_delay"].fillna(delay_impute_value)

    delay_min = float(features["avg_submission_delay"].min())
    delay_max = float(features["avg_submission_delay"].max())
    delay_range = delay_max - delay_min if delay_max != delay_min else 1.0
    features["delay_norm"] = (
        (features["avg_submission_delay"] - delay_min) / delay_range
    ).clip(0, 1)
    features.loc[features["has_submission_by_horizon"].eq(0), "delay_norm"] = 0

    features["assessment_discipline"] = (
        0.5 * features["completion_ratio"] + 0.5 * features["delay_norm"]
    )

    valid_activity_window = (
        features["first_activity_date"].ge(0) & features["last_activity_date"].ge(0)
    )
    features["activity_span"] = 0
    features.loc[valid_activity_window, "activity_span"] = (
        features.loc[valid_activity_window, "last_activity_date"]
        - features.loc[valid_activity_window, "first_activity_date"]
        + 1
    ).clip(lower=1)
    features["persistence_score"] = 0.0
    features.loc[valid_activity_window, "persistence_score"] = (
        features.loc[valid_activity_window, "active_days"]
        / features.loc[valid_activity_window, "activity_span"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)

    eng_norm = _minmax_norm(features["total_clicks"])
    score_norm = _minmax_norm(features["avg_score"])
    early_norm = _minmax_norm(features["early_engagement_ratio"])
    features["learning_risk_index"] = (
        0.25 * (1 - eng_norm)
        + 0.30 * (1 - score_norm)
        + 0.20 * (1 - features["persistence_score"])
        + 0.15 * (1 - features["assessment_discipline"])
        + 0.10 * (1 - early_norm)
    ).round(4)

    fill_defaults = {
        "first_activity_date": -1,
        "last_activity_date": -1,
        "days_since_last": horizon + 1,
        "activity_span": 0,
        "num_total_assessments": 0,
    }
    for col, value in fill_defaults.items():
        features[col] = features[col].fillna(value)

    zero_fill_cols = [
        "total_clicks",
        "active_days",
        "early_engagement",
        "early_engagement_ratio",
        "avg_score",
        "score_std",
        "engagement_intensity",
        "assessment_discipline",
        "persistence_score",
        "learning_risk_index",
        "num_submitted",
        "completion_ratio",
        "delay_norm",
        "has_submission_by_horizon",
    ]
    for col in zero_fill_cols:
        features[col] = features[col].fillna(0)

    for col in ["gender", "region", "highest_education", "imd_band", "age_band", "disability"]:
        features[col] = features[col].fillna("Unknown")

    for col in ["total_clicks", "active_days", "early_engagement", "engagement_intensity"]:
        features[f"{col}_log"] = np.log1p(features[col])

    features["at_risk"] = features["final_result"].isin(AT_RISK_OUTCOMES).astype(int)

    stats = {
        "horizon_day": horizon,
        "rows": int(len(features)),
        "raw_missing_pct_pre_imputation": raw_missing_pct,
        "post_imputation_missing_pct": float(features.isna().mean().mean() * 100),
        "observed_activity_rate": activity_coverage,
        "submission_coverage": float(features["has_submission_by_horizon"].mean()),
        "score_coverage": score_coverage,
        "available_assessment_rate": float(features["num_total_assessments"].gt(0).mean()),
        "delay_imputation_value": delay_impute_value,
        "at_risk_rate": float(features["at_risk"].mean()),
    }
    return features, stats


def _build_prediction_table(
    features: pd.DataFrame,
    categorical_template: pd.DataFrame,
) -> pd.DataFrame:
    prediction = features[KEY + ["horizon_day"] + PRED_NUM_FEATURES + ["at_risk"]].copy()
    prediction = prediction.merge(categorical_template, on=KEY, how="left")
    return prediction


def build_multi_horizon_feature_store(
    processed_dir: Path,
    horizons: tuple[int, ...] = (7, 14, 21, 30),
    early_window_cap: int = 14,
    write_outputs: bool = True,
) -> FeatureStoreOutputs:
    processed_dir = Path(processed_dir)
    tables = _load_core_tables(processed_dir)

    student_info = tables["student_info"]
    categorical_template = _build_categorical_template(student_info)

    vle_features = _build_vle_features_by_horizon(
        tables["student_vle"],
        horizons=horizons,
        early_window_cap=early_window_cap,
    )

    assess = _prepare_assessment_table(
        tables["assessment_performance"],
        tables["assessments_clean"],
    )
    assessment_features, total_assessments = _build_assessment_features_by_horizon(
        assess,
        tables["assessments_clean"],
        horizons=horizons,
    )

    features_by_horizon: dict[int, pd.DataFrame] = {}
    prediction_tables: dict[int, pd.DataFrame] = {}
    metadata_rows: list[dict[str, float]] = []

    for horizon in horizons:
        features, stats = _build_features_for_horizon(
            student_info=student_info,
            vle_features=vle_features[horizon],
            assessment_features=assessment_features[horizon],
            total_assessments=total_assessments[horizon],
            horizon=horizon,
        )
        features_by_horizon[horizon] = features
        prediction_tables[horizon] = _build_prediction_table(features, categorical_template)
        stats["prediction_columns"] = int(prediction_tables[horizon].shape[1])
        stats["numeric_model_features"] = len(PRED_NUM_FEATURES)
        stats["categorical_dummy_features"] = int(categorical_template.shape[1] - len(KEY))
        metadata_rows.append(stats)

    features_horizon_metadata = pd.DataFrame(metadata_rows).sort_values("horizon_day")

    day30_features = features_by_horizon[max(horizons)].copy()
    day30_public = day30_features.rename(
        columns={"has_submission_by_horizon": "has_submission_by_day30"}
    )
    features_final = day30_public[FULL_FEATURE_COLUMNS].copy()

    scaler_seg = StandardScaler()
    features_segmentation = day30_features[KEY + SEG_FEATURES].copy()
    features_segmentation[SEG_FEATURES] = scaler_seg.fit_transform(features_segmentation[SEG_FEATURES])

    scaler_rec = MinMaxScaler()
    features_recommendation = day30_features[KEY + REC_FEATURES + ["final_result", "at_risk"]].copy()
    features_recommendation[REC_FEATURES] = scaler_rec.fit_transform(features_recommendation[REC_FEATURES])

    features_prediction = prediction_tables[max(horizons)].copy()

    if write_outputs:
        features_final.to_csv(processed_dir / "features_final.csv", index=False)
        features_segmentation.to_csv(processed_dir / "features_segmentation.csv", index=False)
        features_prediction.to_csv(processed_dir / "features_prediction.csv", index=False)
        features_recommendation.to_csv(processed_dir / "features_recommendation.csv", index=False)
        features_horizon_metadata.to_csv(processed_dir / "features_horizon_metadata.csv", index=False)
        for horizon, prediction_table in prediction_tables.items():
            prediction_table.to_csv(
                processed_dir / f"features_prediction_day{horizon:02d}.csv",
                index=False,
            )

    return FeatureStoreOutputs(
        features_by_horizon=features_by_horizon,
        prediction_tables=prediction_tables,
        features_horizon_metadata=features_horizon_metadata,
        features_final=features_final,
        features_segmentation=features_segmentation,
        features_prediction=features_prediction,
        features_recommendation=features_recommendation,
    )
