# Power BI Dashboard Storyboard

This file is the dashboard build guide for the final project.
It maps each page to:

- business question,
- core visuals,
- KPI cards,
- source tables,
- interaction logic,
- short speaking notes.

Use the exported CSV files from notebooks `05` and `06` together with the processed tables from notebooks `02` to `04`.

## 1. Recommended Data Imports

Import these files into Power BI:

- `student_info_clean.csv`
- `student_course_registration.csv`
- `assessment_performance.csv`
- `student_vle_summary.csv`
- `features_final.csv`
- `segment_assignments.csv`
- `cluster_profiles.csv`
- `cluster_success_activity_profile.csv`
- `personalized_learning_paths.csv`
- `at_risk_test_predictions.csv`
- `segment_model_performance.csv`
- `recommendation_dashboard_summary.csv`
- `outcome_risk_summary.csv`

## 2. Suggested Relationship Logic

Use enrollment grain as the main key:

- `id_student`
- `code_module`
- `code_presentation`

Primary fact-like tables:

- `features_final`
- `personalized_learning_paths`
- `at_risk_test_predictions`

Profile / summary tables:

- `segment_assignments`
- `cluster_profiles`
- `cluster_success_activity_profile`
- `segment_model_performance`
- `outcome_risk_summary`

## 3. Suggested Core Measures

Use DAX or equivalent measures like these:

```text
Total Enrollments = COUNTROWS(features_final)
Unique Learners = DISTINCTCOUNT(features_final[id_student])
At-Risk Learners = CALCULATE(COUNTROWS(features_final), features_final[at_risk] = 1)
At-Risk Rate = DIVIDE([At-Risk Learners], [Total Enrollments])
Avg Score = AVERAGE(features_final[avg_score])
Avg Completion Ratio = AVERAGE(features_final[completion_ratio])
Avg Recommendation Score = AVERAGE(personalized_learning_paths[recommendation_score])
Predicted Positive Rate = AVERAGE(at_risk_test_predictions[y_pred])
Avg Risk Probability = AVERAGE(at_risk_test_predictions[risk_probability])
```

## 4. Page 1 - Executive Overview

### Business purpose

Give a top-level summary of learner risk, engagement, segmentation, and recommendation status.

### KPI cards

- Total Enrollments: `32,593`
- Unique Learners: `28,785`
- At-Risk Rate: `52.8%`
- Avg Recommendation Score
- Number of Segments: `4`
- Champion Model Recall: `91.55%`

### Suggested visuals

1. KPI cards row
2. Final result distribution donut or stacked bar
3. At-risk rate by module bar chart
4. Cluster distribution bar chart
5. Recommended path distribution bar chart

### Source tables

- `features_final`
- `segment_assignments`
- `personalized_learning_paths`
- `cluster_profiles`

### Key message to say

> The project solves a real risk problem at scale. More than half of enrollments are at risk, and learner behavior can be translated into both segment insight and intervention paths.

## 5. Page 2 - Learning Behavior Analytics

### Business purpose

Show how engagement and assessment behavior differ across learner outcomes.

### KPI cards

- Median total clicks
- Median early engagement
- Avg score
- Avg completion ratio

### Suggested visuals

1. Boxplot or violin chart of `total_clicks` by `final_result`
2. Boxplot of `avg_score` by `final_result`
3. Boxplot of `completion_ratio` by `final_result`
4. Bar chart of median early engagement by outcome
5. Heatmap of modules x at-risk rate

### Important story points

- `Withdrawn` learners show the weakest early engagement.
- Assessment completion and score quality separate good and weak outcomes sharply.
- Module context matters; `CCC` is much riskier than `AAA`.

### Source tables

- `features_final`

## 6. Page 3 - Personalized Recommendation for General Learners

### Business purpose

Explain learner segmentation and the recommendation engine for general learners.

### KPI cards

- Number of learners in each segment
- Avg recommendation score
- Top recommended path

### Suggested visuals

1. Cluster size bar chart
2. Cluster profile matrix or heatmap
3. Recommended path distribution
4. Activity-type mix by cluster
5. Table of sample personalized recommendations:
   - `cluster_label`
   - `recommended_path`
   - `action_1`
   - `activity_recommendation_1`
   - `recommendation_score`

### Source tables

- `cluster_profiles`
- `cluster_success_activity_profile`
- `personalized_learning_paths`
- `segment_assignments`

### Key message to say

> Recommendation is driven by behavior gaps relative to successful peers in the same cluster, not by one-size-fits-all advice.

## 7. Page 4 - At-Risk Learner Identification

### Business purpose

Show model performance and how the early-warning model flags risk.

### KPI cards

- Champion model: `Random Forest`
- Threshold: `0.30`
- Recall: `91.55%`
- Precision: `65.32%`
- ROC-AUC: `0.8427`
- PR-AUC: `0.8720`

### Suggested visuals

1. Model comparison table:
   - model
   - threshold
   - precision
   - recall
   - F2
   - ROC-AUC
   - PR-AUC
2. Confusion matrix card block
3. ROC curve image or recreated curve
4. Precision-recall curve
5. Histogram of `risk_probability`

### Source tables

- `model_test_comparison`
- `at_risk_test_predictions`

### Key message to say

> The final operating point intentionally prioritizes recall because missing an at-risk learner is more costly than reviewing an extra false alarm.

## 8. Page 5 - At-Risk Behavior Deep Dive

### Business purpose

Explain where the model works best and what behavioral signals are most important.

### KPI cards

- Top risk driver: `avg_score`
- Strongest operational group: `Inactive Drop-offs`
- Highest predicted-positive group: `Inactive Drop-offs`

### Suggested visuals

1. Feature importance bar chart from `model_feature_importance`
2. Segment-level model performance table
3. Risk probability by final result boxplot
4. Predicted positive rate by cluster
5. Actual at-risk rate by cluster

### Source tables

- `model_feature_importance`
- `segment_model_performance`
- `outcome_risk_summary`
- `at_risk_test_predictions`

### What to explain

- The model is strongest in clearly vulnerable segments.
- `Inactive Drop-offs` and `Sporadic Explorers` should be treated as high-priority intervention groups.
- Risk is shaped by both engagement and academic performance, not by one metric alone.

## 9. Page 6 - Personalized Path Recommendation for At-Risk Learners

### Business purpose

Connect prediction to action: once a learner is flagged at risk, what should be recommended next?

### Filters

- `y_pred = 1`
- or `risk_probability >= chosen managerial cutoff`

### KPI cards

- Number of flagged learners
- Avg risk probability among flagged learners
- Top recommended path for flagged learners
- Avg recommendation score for flagged learners

### Suggested visuals

1. Bar chart of recommended paths among predicted at-risk learners
2. Cluster x recommended path matrix
3. Table of flagged learners with:
   - `id_student`
   - `code_module`
   - `cluster_label`
   - `risk_probability`
   - `recommended_path`
   - `action_1`
   - `action_2`
   - `activity_recommendation_1`
4. Segment-wise intervention table:
   - cluster
   - actual at-risk rate
   - recall
   - most common recommended path

### Source tables

- `at_risk_test_predictions`
- `personalized_learning_paths`
- `segment_model_performance`

### Key message to say

> The project does not stop at prediction. It links risk detection to a concrete intervention path, which is the practical value of the full pipeline.

## 10. Recommended Slicers

Use these slicers consistently across pages:

- `code_module`
- `code_presentation`
- `cluster_label`
- `rule_segment`
- `final_result`
- `recommended_path`

Optional:

- risk flag (`y_pred`)
- high-risk band based on `risk_probability`

## 11. Design Notes

Keep the dashboard style quiet and analytical:

- avoid decorative visuals,
- favor readable KPI cards and bar charts,
- keep one primary message per page,
- use consistent colors for outcomes and clusters,
- keep filters in a fixed location.

Suggested color logic:

- `Distinction`: blue
- `Pass`: green
- `Fail`: orange
- `Withdrawn`: red
- risk KPI / predicted positive: red family
- recommendation / action visuals: teal or blue-green family

## 12. Final Presentation Flow

If you need a clean spoken sequence while presenting the dashboard:

1. Start with scale and risk in the Executive Overview.
2. Show that weak outcomes are already visible in early behavior.
3. Move to segmentation and explain that learners are not homogeneous.
4. Show the at-risk model and justify the threshold with recall.
5. End with intervention: recommendation path by flagged learner group.

That sequence is usually the easiest to defend because it moves from problem -> evidence -> segmentation -> prediction -> action.
