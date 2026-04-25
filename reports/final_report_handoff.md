# Final Report Handoff

This document turns notebooks `01` to `06` into a report-ready storyline.
It is written to help with:

- final report drafting,
- presentation speaking flow,
- defense Q&A on methods and model choice.

## 1. Executive Summary

Use this as the opening paragraph of the report:

> This project analyzes learning behavior in the OULAD dataset and develops a data-driven framework for learner segmentation, personalized learning path recommendation, and early identification of at-risk learners. Across 32,593 enrollments and 28,785 unique learners, the analysis shows that early engagement, assessment completion, persistence, and study discipline are much stronger outcome signals than demographic variables alone. The project then translates these signals into four learner segments, personalized recommendation paths, and an early-warning at-risk model designed for high recall so that weak learners can be detected early enough for intervention.

## 2. Project Scope and Objective

Use these points in the introduction:

- Dataset: OULAD with seven main tables.
- Observation unit: learner-module-presentation enrollment.
- Part I objective: understand learning behavior, segment learners, and recommend learning paths.
- Part II objective: identify at-risk learners early and provide intervention-oriented insight.

## 3. Dataset and Preparation

Key facts to state:

- Total enrollments used in the final pipeline: `32,593`
- Unique learners: `28,785`
- Final result distribution:
  - `Pass`: `12,361`
  - `Withdrawn`: `10,156`
  - `Fail`: `7,052`
  - `Distinction`: `3,024`
- At-risk definition: `Fail` or `Withdrawn`
- Overall at-risk rate: `52.8%`

Suggested wording:

> Data preparation was carried out at the enrollment level to avoid unsafe student-only joins. Missing and inconsistent fields were audited, cleaned tables were exported, and all downstream features were rebuilt from a consistent learner-module-presentation grain.

## 4. EDA Storyline

This should be the report flow for notebook `03_eda.ipynb`.

### 4.1 Overall risk context

- The at-risk population is large: `52.8%` of enrollments.
- Risk is not evenly distributed across courses.
- Highest-risk module: `CCC` with `62.2%` at-risk rate.
- Lowest-risk module: `AAA` with `29.0%` at-risk rate.

Suggested sentence:

> The risk problem is operationally meaningful, not marginal. More than half of enrollments fall into fail or withdrawal outcomes, and the risk burden varies substantially across modules.

### 4.2 Early engagement matters

Median early clicks in the first 14 days:

- `Distinction`: `156`
- `Pass`: `104`
- `Fail`: `40`
- `Withdrawn`: `12`

Suggested sentence:

> The early-window engagement gap appears immediately. Learners who eventually withdraw show extremely weak participation even in the first two weeks, while successful learners build momentum much earlier.

### 4.3 Assessment behavior matters

Median completion ratio:

- `Distinction`: `0.5`
- `Pass`: `0.5`
- `Fail`: `0.5`
- `Withdrawn`: `0.0`

Median average score:

- `Distinction`: `86`
- `Pass`: `75`
- `Fail`: `55`
- `Withdrawn`: `0`

Suggested sentence:

> Assessment completion and score quality sharply separate successful learners from withdrawn learners. This confirms that academic discipline and not only browsing intensity should be central in downstream modeling and recommendation design.

## 5. Feature Engineering Summary

This is the logic to describe notebook `04_feature_engineering.ipynb`.

### 5.1 Day-30 feature design

State clearly:

- All predictive features were restricted to information available by day 30.
- This was done to reduce future-information leakage.
- Features were grouped into:
  - engagement,
  - timing,
  - assessment,
  - persistence,
  - composite risk and discipline features.

### 5.2 Important engineered variables

Name these explicitly:

- `total_clicks_log`
- `active_days_log`
- `early_engagement_ratio`
- `days_since_last`
- `avg_score`
- `avg_submission_delay`
- `completion_ratio`
- `assessment_discipline`
- `persistence_score`
- `learning_risk_index`
- `has_submission_by_day30`

Suggested sentence:

> The feature store was designed to support both unsupervised segmentation and supervised early-warning prediction while preserving a strict day-30 observation window.

## 6. Segmentation and Recommendation Story

Use this to explain notebook `05_segmentation_recommendation.ipynb`.

### 6.1 Why both rule-based and K-Means segmentation were used

Suggested wording:

> A rule-based segmentation was built first to preserve business interpretability, then K-Means clustering was used to discover cleaner multivariate learner profiles from the standardized behavioral space.

### 6.2 Final segments

Report these four clusters:

1. `Inactive Drop-offs`
2. `Sporadic Explorers`
3. `Steady Progressors`
4. `Focused Achievers`

Cluster summary:

| Cluster | Enrollments | Share | At-risk rate | Median clicks | Median completion | Median avg score |
|---|---:|---:|---:|---:|---:|---:|
| Inactive Drop-offs | 5,018 | 15.4% | 93.08% | 0 | 0.0 | 0 |
| Sporadic Explorers | 8,476 | 26.01% | 60.82% | 79 | 0.0 | 0 |
| Steady Progressors | 17,047 | 52.30% | 39.43% | 241 | 0.5 | 78 |
| Focused Achievers | 2,052 | 6.30% | 32.16% | 541 | 1.0 | 80 |

Interpretation:

- `Inactive Drop-offs` are the clearest intervention group.
- `Sporadic Explorers` engage a little, but do not convert activity into completion.
- `Steady Progressors` are the operational middle and need guidance, not rescue.
- `Focused Achievers` can be treated as a benchmark group and recommendation prototype source.

### 6.3 Recommendation layer

Core logic:

- compare each learner with successful peers in the same cluster,
- derive the largest feature gap,
- assign a learning path label,
- recommend high-value activity types available in the learner's module.

Top path distribution:

- `Consistency Building Path`: `11,133`
- `Early Start Path`: `7,108`
- `Assessment Recovery Path`: `5,613`
- `Mastery Improvement Path`: `5,145`
- `Re-Engagement Path`: `1,681`
- `Focused Study Path`: `1,095`
- `Sustain & Deepen Path`: `569`
- `Assessment Discipline Path`: `249`

Suggested sentence:

> The recommendation engine does not simply rank content. It translates the behavioral gap between a learner and successful peers into an actionable path, concrete next actions, and activity-type emphasis.

## 7. At-Risk Modeling Story

Use this for notebook `06_at_risk_modeling.ipynb`.

### 7.1 Candidate models

State that three models were compared:

- Logistic Regression
- Random Forest
- XGBoost

### 7.2 Why threshold tuning was necessary

Suggested wording:

> Because the project goal is early intervention, the model was not evaluated only at the default threshold of 0.50. Instead, the validation split was used to choose an operating threshold that preserves high recall while improving precision as much as possible.

### 7.3 Model selection logic

State this clearly:

- Use validation data, not test data, to choose the final operating point.
- Require recall to stay around or above `0.90`.
- Among such thresholds, prefer higher precision.

This is the strongest defense point if asked why Random Forest was selected.

### 7.4 Validation operating points

| Model | Threshold | Precision | Recall | F2 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.30 | 0.6436 | 0.9024 | 0.8352 | 0.8217 | 0.8562 |
| XGBoost | 0.25 | 0.6303 | 0.9227 | 0.8444 | 0.8293 | 0.8640 |
| Logistic Regression | 0.25 | 0.6197 | 0.9137 | 0.8345 | 0.7956 | 0.8289 |

### 7.5 Final test performance

Champion model used in the notebook: `Random Forest`

| Metric | Value |
|---|---:|
| Threshold | `0.30` |
| Accuracy | `0.6987` |
| Precision | `0.6532` |
| Recall | `0.9155` |
| F1 | `0.7624` |
| F2 | `0.8474` |
| ROC-AUC | `0.8427` |
| PR-AUC | `0.8720` |

Suggested model-choice explanation:

> Random Forest was retained as the final model because it provided the strongest precision among the validation-selected high-recall operating points. XGBoost remained a strong benchmark and slightly exceeded Random Forest on some held-out ranking metrics, but it was not selected post hoc from the test set in order to preserve methodological discipline.

This sentence is important. It shows you did not overfit your decision to test performance.

### 7.6 Main predictive signals

Top importance signals to mention:

1. `avg_score`
2. `studied_credits`
3. `days_since_last`
4. `avg_submission_delay`
5. `learning_risk_index`
6. `completion_ratio`
7. `assessment_discipline`
8. `active_days_log`
9. `total_clicks_log`

Interpretation:

- academic quality and engagement timing matter together,
- disengagement recency remains critical,
- assessment behavior is central in both recommendation and prediction.

### 7.7 Cluster-level model behavior

Use these lines in the discussion:

- `Inactive Drop-offs`: actual at-risk rate `93.0%`, recall `0.9989`
- `Sporadic Explorers`: actual at-risk rate `61.56%`, recall `0.9807`
- `Steady Progressors`: actual at-risk rate `38.85%`, recall `0.8289`
- `Focused Achievers`: actual at-risk rate `33.49%`, recall `0.6879`

Suggested interpretation:

> The model is strongest in the clearly vulnerable groups and naturally weaker in the more ambiguous high-functioning groups. This is acceptable for an early-warning system whose operational objective is to catch disengagement early rather than to maximize precision on already stable learners.

## 8. Managerial Implications

Use this as the final practical section.

### 8.1 For instructors

- Monitor low-engagement and low-submission learners before the course reaches day 30.
- Treat `Inactive Drop-offs` and `Sporadic Explorers` as intervention-priority groups.
- Use recommendation paths to tailor follow-up actions instead of giving the same advice to everyone.

### 8.2 For program managers

- Compare risk across modules, especially `CCC` versus `AAA`.
- Use at-risk heatmaps and segment dashboards to allocate support resources.
- Combine model prediction with learner segment to prioritize who needs structured intervention versus who only needs guidance.

### 8.3 For platform design

- Push earlier engagement prompts in the first two weeks.
- Increase visibility of assessment-related actions.
- Emphasize high-value activity types associated with successful peers.

## 9. Limitations

Use these in the limitations section:

- The target is based on final outcomes and uses `Fail + Withdrawn` as a broad at-risk definition.
- Recommendations are rule-based and prototype-based, not causal estimates.
- The model is built from one dataset and should be externally validated before operational deployment.
- Some demographic effects may reflect contextual rather than causal relationships.

## 10. Suggested Report Structure

Use this chapter order:

1. Introduction
2. Research objectives and questions
3. Dataset and methodology
4. Data cleaning and integration
5. Exploratory data analysis
6. Feature engineering
7. Learner segmentation and personalized recommendation
8. At-risk learner modeling
9. Managerial implications
10. Limitations and future work
11. Conclusion

## 11. Defense Q&A Short Answers

### Why did you not use threshold 0.50?

Because this is an early-warning problem. Missing an at-risk learner is more costly than reviewing an extra flagged learner, so the threshold was selected on validation data to preserve recall above about 0.90.

### Why did you choose Random Forest instead of XGBoost?

Because the final choice was made from validation performance under the project decision rule, not from test-set hindsight. Random Forest delivered the best precision among the high-recall validation operating points.

### Why include segmentation before prediction?

Because segmentation helps explain heterogeneity in learner behavior and makes recommendations more actionable. It turns model output into intervention logic.

### Why use both rule-based and K-Means segmentation?

The rule-based layer gives business interpretability, while K-Means provides a more data-driven multivariate grouping. Using both improves credibility and communication.
