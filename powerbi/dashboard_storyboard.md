# Power BI Research Dashboard Storyboard

This storyboard rebuilds the dashboard around the upgraded project direction:

`behavior evidence -> segmentation context -> multi-horizon early warning -> calibration -> watchlist`

The dashboard is no longer framed as a product demo. It is a **research dashboard** that supports evidence, explanation, and intervention prioritization.

## 1. Data Imports

Import these tables into Power BI:

- `features_final.csv`
- `features_horizon_metadata.csv`
- `segment_assignments.csv`
- `cluster_profiles.csv`
- `cluster_success_activity_profile.csv`
- `personalized_learning_paths.csv`
- `recommendation_dashboard_summary.csv`
- `model_horizon_comparison.csv`
- `selected_operating_points.csv`
- `champion_test_metrics.csv`
- `champion_test_predictions.csv`
- `calibration_summary.csv`
- `risk_band_summary.csv`
- `risk_band_test_predictions.csv`
- `ablation_results.csv`
- `ablation_gain_summary.csv`
- `model_feature_importance.csv`
- `segment_model_performance.csv`
- `outcome_risk_summary.csv`

## 2. Relationship Logic

Use the enrollment key everywhere:

- `id_student`
- `code_module`
- `code_presentation`

Primary fact-like tables:

- `features_final`
- `personalized_learning_paths`
- `champion_test_predictions`
- `risk_band_test_predictions`

Profile / summary tables:

- `features_horizon_metadata`
- `cluster_profiles`
- `cluster_success_activity_profile`
- `model_horizon_comparison`
- `selected_operating_points`
- `champion_test_metrics`
- `calibration_summary`
- `risk_band_summary`
- `ablation_results`
- `ablation_gain_summary`
- `model_feature_importance`
- `segment_model_performance`
- `outcome_risk_summary`

## 3. Core Measures

Use DAX measures like these:

```text
Total Enrollments = COUNTROWS(features_final)
Unique Learners = DISTINCTCOUNT(features_final[id_student])
At-Risk Learners = CALCULATE(COUNTROWS(features_final), features_final[at_risk] = 1)
At-Risk Rate = DIVIDE([At-Risk Learners], [Total Enrollments])

Champion Precision = MAX(champion_test_metrics[precision])
Champion Recall = MAX(champion_test_metrics[recall])
Champion PR AUC = MAX(champion_test_metrics[pr_auc])
Champion ROC AUC = MAX(champion_test_metrics[roc_auc])

Avg Risk Probability = AVERAGE(champion_test_predictions[risk_probability])
Critical Risk Learners = CALCULATE(COUNTROWS(risk_band_test_predictions), risk_band_test_predictions[risk_band] = "Critical")
Earliest Useful Horizon = MINX(FILTER(selected_operating_points, selected_operating_points[is_earliest_useful_horizon] = TRUE()), selected_operating_points[horizon_day])
```

## 4. Page 1 - Executive Overview

### Business purpose

Show the project scale, risk burden, final champion result, and overall intervention context.

### KPI cards

- Total enrollments: `32,593`
- Unique learners: `28,785`
- At-risk rate: `52.8%`
- Earliest useful horizon: `Day 7`
- Champion pair: `XGBoost @ Day 30`
- Champion recall: `93.67%`
- Champion PR-AUC: `0.8766`

### Suggested visuals

1. KPI cards row
2. Final result distribution bar chart
3. Segment distribution bar chart
4. Top recommendation paths bar chart
5. Risk band distribution chart

### Source tables

- `features_final`
- `segment_assignments`
- `personalized_learning_paths`
- `risk_band_summary`

### Key message

> The project addresses a large risk problem, keeps the segmentation layer for interpretation, and upgrades the predictive branch into a calibrated multi-horizon early-warning study.

## 5. Page 2 - Behavior & Outcome Analytics

### Business purpose

Show why early warning is justified by the raw behavior patterns.

### KPI cards

- Median early engagement
- Avg score
- Avg completion ratio
- Avg days since last activity

### Suggested visuals

1. Boxplot of `total_clicks` by `final_result`
2. Boxplot of `avg_score` by `final_result`
3. Boxplot of `completion_ratio` by `final_result`
4. Median early engagement by outcome
5. Module x at-risk rate heatmap

### Source tables

- `features_final`

### Key message

> Weak outcomes are visible early and are shaped by both engagement and assessment behavior.

## 6. Page 3 - Segmentation & Recommendation Context

### Business purpose

Keep the day-30 learner analytics story visible and connect it to recommendation logic.

### KPI cards

- Number of segments: `4`
- Largest segment: `Steady Progressors`
- Highest-risk segment: `Inactive Drop-offs`
- Avg recommendation score

### Suggested visuals

1. Cluster size bar chart
2. Cluster risk-rate bar chart
3. Cluster profile heatmap
4. Recommended path distribution
5. Sample recommendation table:
   - `cluster_label`
   - `recommended_path`
   - `action_1`
   - `action_2`
   - `recommendation_score`

### Source tables

- `segment_assignments`
- `cluster_profiles`
- `cluster_success_activity_profile`
- `personalized_learning_paths`

### Key message

> Segmentation remains the interpretation layer that turns model output into differentiated intervention ideas.

## 7. Page 4 - Multi-Horizon Early Warning

### Business purpose

Compare how predictive quality changes from day 7 to day 30.

### KPI cards

- Earliest useful horizon: `Day 7`
- Best day-7 recall: `93.93%`
- Best day-30 PR-AUC: `0.8639` on validation
- Final champion threshold: `0.25`

### Suggested visuals

1. Matrix:
   - horizon
   - model
   - validation precision
   - validation recall
   - validation F2
   - validation PR-AUC
2. Line chart of best validation PR-AUC by horizon
3. Line chart of best validation recall by horizon
4. Threshold curve for the champion pair
5. Card or annotation that explains:
   - day 7 = earliest useful horizon
   - day 30 = strongest final ranking horizon

### Source tables

- `model_horizon_comparison`
- `selected_operating_points`
- `threshold_search_by_horizon`
- `champion_test_metrics`

### Key message

> Earlier horizons lose information but gain intervention time. Later horizons rank learners better, but the day-7 checkpoint is already operationally useful.

## 8. Page 5 - Calibration & Model Diagnostics

### Business purpose

Show whether the final probabilities are trustworthy and which features matter most.

### KPI cards

- Champion PR-AUC: `0.8766`
- Champion ROC-AUC: `0.8467`
- Champion Brier score: `0.1580`
- Critical-band realized at-risk rate: `92.5%`

### Suggested visuals

1. Calibration line chart across horizons for the champion model family
2. Risk band summary chart
3. Ablation comparison bar chart at day 14
4. Ablation comparison bar chart at day 30
5. Feature importance bar chart

### Source tables

- `calibration_summary`
- `risk_band_summary`
- `ablation_results`
- `ablation_gain_summary`
- `model_feature_importance`

### Key message

> The probabilities are ordered well enough to support risk tiers, and the strongest lift comes from combining engagement and assessment features rather than relying on demographics alone.

## 9. Page 6 - At-Risk Learner Watchlist

### Business purpose

Turn the research output into a prioritized learner table for review.

### KPI cards

- Number of predicted positives
- Number of critical-risk learners
- Avg risk probability among predicted positives
- Most common recommendation path among flagged learners

### Suggested visuals

1. Watchlist table with:
   - `id_student`
   - `code_module`
   - `cluster_label`
   - `risk_band`
   - `risk_probability`
   - `recommended_path`
   - `action_1`
   - `action_2`
2. Bar chart of flagged learners by risk band
3. Cluster x risk band matrix
4. Recommended path distribution for flagged learners

### Source tables

- `champion_test_predictions`
- `risk_band_test_predictions`
- `segment_model_performance`

### Key message

> The dashboard does not stop at prediction. It organizes the final output into a triage list that combines risk level, learner segment, and recommended path.

## 10. Recommended Slicers

Use these slicers consistently:

- `code_module`
- `code_presentation`
- `cluster_label`
- `rule_segment`
- `final_result`
- `risk_band`
- `recommended_path`

## 11. Design Notes

Keep the dashboard analytical:

- restrained layout
- no decorative hero sections
- fixed KPI row on top
- consistent red family for risk
- teal or blue-green for recommendation visuals
- one key message per page

Suggested outcome colors:

- `Distinction`: blue
- `Pass`: green
- `Fail`: orange
- `Withdrawn`: red

## 12. Presentation Flow

Use this order while presenting:

1. Executive scale and why the problem matters
2. Behavior evidence
3. Segmentation context
4. Multi-horizon comparison
5. Calibration and ablation
6. Watchlist and intervention relevance
