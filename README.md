# Learning Behavior Analytics and Multi-Horizon Early Warning for At-Risk Learners

This project uses the OULAD dataset to build an end-to-end learning analytics pipeline at the `learner + module + presentation` grain. The final deliverable is no longer just a day-30 classifier. It is a **research-focused early-warning system** with multi-horizon feature engineering, segmentation, recommendation, ablation analysis, calibration analysis, and Power BI handoff artifacts.

## What Makes This Version Different

The project now answers four stronger questions:

1. How early can at-risk learners be identified at `day 7`, `14`, `21`, and `30`?
2. Which horizon-model pair is strongest once recall-oriented thresholding is enforced?
3. Which feature groups actually create the predictive lift?
4. Are the predicted probabilities reliable enough to support risk-banded intervention?

## Key Results

- Final analytic sample: `32,593` enrollments and `28,785` unique learners
- At-risk definition: `Fail` or `Withdrawn`
- Overall at-risk rate: `52.8%`
- Earliest useful horizon: `day 7`
  - best day-7 operating point: `Logistic Regression`
  - validation recall: `0.9393`
  - validation precision: `0.5771`
  - validation PR-AUC: `0.7592`
- Final champion pair: `XGBoost @ day 30`, threshold `0.25`
  - test accuracy: `0.6777`
  - test precision: `0.6313`
  - test recall: `0.9367`
  - test F2: `0.8540`
  - test ROC-AUC: `0.8467`
  - test PR-AUC: `0.8766`
- Final risk bands on the held-out test set:
  - `Low`: actual at-risk rate `15.4%`
  - `Medium`: `35.3%`
  - `High`: `62.2%`
  - `Critical`: `92.5%`

## Segmentation and Recommendation Highlights

Notebook `05` still keeps the project's interpretable learner analytics layer:

- `Inactive Drop-offs`
- `Sporadic Explorers`
- `Steady Progressors`
- `Focused Achievers`

Recommendation paths are generated from behavioral gaps relative to successful peers in each cluster. The most common paths are:

- `Consistency Building Path`: `11,133`
- `Early Start Path`: `7,108`
- `Assessment Recovery Path`: `5,613`
- `Mastery Improvement Path`: `5,145`

## Research Contributions Added in This Upgrade

### 1. Multi-horizon feature store

Notebook `04` now exports:

- `features_prediction_day07.csv`
- `features_prediction_day14.csv`
- `features_prediction_day21.csv`
- `features_prediction_day30.csv`
- `features_horizon_metadata.csv`

All four tables:

- have `32,593` rows,
- have consistent schemas with `40` columns,
- have `0` missing cells,
- have `0` duplicate enrollment keys.

### 2. Multi-horizon modeling

Notebook `06` now compares:

- `Logistic Regression`
- `Random Forest`
- `XGBoost`

across:

- `day 7`
- `day 14`
- `day 21`
- `day 30`

with a fixed validation rule:

1. keep thresholds with `recall >= 0.90`
2. maximize `precision`
3. break ties with `F2`, then `PR-AUC`

### 3. Ablation study

Ablation is run at `day 14` and `day 30` with five feature groups:

- `demographics only`
- `engagement only`
- `assessment only`
- `engagement + assessment`
- `full feature set`

The strongest pattern is consistent: **full feature sets outperform simpler baselines**, and `engagement + assessment` is clearly stronger than demographics alone.

### 4. Calibration and risk bands

The project now exports:

- `calibration_summary.csv`
- `risk_band_summary.csv`
- `risk_band_test_predictions.csv`

This makes the system more useful operationally because intervention prioritization depends on probability quality, not only on binary flags.

## Repository Map

- `notebooks/01_*.ipynb` to `06_*.ipynb`: full analysis pipeline
- `src/features/multi_horizon_feature_store.py`: reusable horizon feature engineering logic
- `src/models/multi_horizon_early_warning.py`: reusable multi-horizon modeling logic
- `data/processed/`: all exported modeling, recommendation, and dashboard tables
- `reports/final_report_handoff.md`: report-ready storyline and defense notes
- `reports/demo_talk_track.md`: short demo script
- `reports/final_presentation_outline.md`: final slide structure
- `powerbi/dashboard_storyboard.md`: research dashboard page plan
- `powerbi/dashboard_data_dictionary.md`: dashboard source and field guide
- `powerbi/dashboard_screenshot_pack/`: research dashboard wireframes

## Main Exported Artifacts

### Feature engineering

- `features_final.csv`
- `features_segmentation.csv`
- `features_prediction.csv`
- `features_recommendation.csv`
- `features_prediction_day07.csv`
- `features_prediction_day14.csv`
- `features_prediction_day21.csv`
- `features_prediction_day30.csv`
- `features_horizon_metadata.csv`

### Segmentation and recommendation

- `segment_assignments.csv`
- `cluster_profiles.csv`
- `cluster_success_activity_profile.csv`
- `personalized_learning_paths.csv`
- `recommendation_dashboard_summary.csv`

### Modeling

- `model_horizon_comparison.csv`
- `threshold_search_by_horizon.csv`
- `selected_operating_points.csv`
- `champion_test_predictions.csv`
- `champion_test_metrics.csv`
- `ablation_results.csv`
- `ablation_gain_summary.csv`
- `calibration_summary.csv`
- `risk_band_summary.csv`
- `risk_band_test_predictions.csv`
- `model_feature_importance.csv`
- `error_analysis_samples.csv`
- `segment_model_performance.csv`
- `outcome_risk_summary.csv`

## How to Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pull the full raw dataset

This repository uses Git LFS for `data/raw/studentVle.csv`.

```bash
git lfs install
git lfs pull
```

### 3. Execute notebooks in order

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_understanding.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_data_cleaning_integration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_feature_engineering.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_segmentation_recommendation.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_at_risk_modeling.ipynb
```

### 4. Use the hardened CLI targets

```bash
make feature-store
make modeling
make research
make validate
make test
make test-notebooks
```

- `make validate` checks the exported research artifacts against the acceptance criteria
- `make test` runs the unit and acceptance tests
- `make test-notebooks` runs notebook smoke tests for notebooks `04` and `06`

## Power BI Status

The repository includes the full research dashboard handoff:

- updated storyboard,
- data dictionary,
- page-level wireframes in `powerbi/dashboard_screenshot_pack/`,
- reporting-ready CSV outputs.

The `.pbix` file itself must be assembled in Power BI Desktop from these exported tables.

## Why This Project Is Special

Most student projects stop at a single static classifier. This one now gives a full decision-relevant chain:

`behavior -> segmentation -> recommendation -> multi-horizon early warning -> calibration -> risk bands`

That makes it defensible as both:

- an academic analytics project, and
- a portfolio-ready decision-support case study.

## License

This repository is for academic and educational use.
