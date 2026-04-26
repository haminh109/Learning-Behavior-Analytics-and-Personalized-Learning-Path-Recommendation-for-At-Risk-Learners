# Final Presentation Outline

## Slide 1. Title

- Project title
- Team / course / university
- One-line value proposition:
  - `Behavior analytics + segmentation + multi-horizon early warning for at-risk learners`

## Slide 2. Problem Context

- Why online learner risk matters
- At-risk definition: `Fail` or `Withdrawn`
- Dataset scale:
  - `32,593` enrollments
  - `28,785` learners
  - `52.8%` at-risk rate

## Slide 3. Research Questions

- How do learning behaviors differ by outcome?
- Can learners be segmented into meaningful profiles?
- How early can at-risk learners be identified?
- Which feature groups matter most?
- Are the predicted probabilities useful for intervention prioritization?

## Slide 4. Pipeline Overview

- Data cleaning and integration
- EDA
- Multi-horizon feature engineering
- Segmentation and recommendation
- Multi-horizon modeling
- Ablation and calibration
- Power BI research dashboard

## Slide 5. EDA Insights

- Early engagement gap by outcome
- Assessment completion and score separation
- Module-level risk variation

Key sentence:

> Risk is visible early, and both engagement and assessment behavior matter.

## Slide 6. Segmentation Results

- Four clusters:
  - Inactive Drop-offs
  - Sporadic Explorers
  - Steady Progressors
  - Focused Achievers
- Show cluster size and at-risk rate

## Slide 7. Recommendation Layer

- Recommendation logic:
  - compare with successful peers in the same cluster
  - identify main behavioral gap
  - assign learning path
- Show top recommended paths

## Slide 8. Multi-Horizon Feature Engineering

- Horizons:
  - day 7
  - day 14
  - day 21
  - day 30
- Stress leakage control
- Show coverage growth across horizons

## Slide 9. Model Design

- Candidate models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Train / validation / test split
- Threshold tuning rule:
  - recall first
  - then precision
  - then F2
  - then PR-AUC

## Slide 10. Horizon Comparison

- Table with best pair per horizon
- Key message:
  - day 7 is the earliest useful horizon
  - day 30 is the strongest overall horizon

## Slide 11. Final Champion Model

- Champion: `XGBoost @ day 30`
- Threshold: `0.25`
- Test metrics:
  - Precision `0.6313`
  - Recall `0.9367`
  - F2 `0.8540`
  - ROC-AUC `0.8467`
  - PR-AUC `0.8766`

## Slide 12. Ablation Results

- Compare:
  - demographics only
  - engagement only
  - assessment only
  - engagement + assessment
  - full feature set
- Main message:
  - full feature set is strongest
  - demographics alone are weak

## Slide 13. Calibration and Risk Bands

- Show calibration curve
- Show risk bands:
  - Low
  - Medium
  - High
  - Critical
- Main message:
  - model probabilities are usable for tiered intervention

## Slide 14. Diagnostic Interpretation

- Top feature importance:
  - avg_score
  - studied_credits
  - days_since_last
  - avg_submission_delay
  - assessment_discipline
- Cluster-level recall differences

## Slide 15. Power BI Research Dashboard

- 6 pages:
  - Executive Overview
  - Behavior & Outcome Analytics
  - Segmentation & Recommendation Context
  - Multi-Horizon Early Warning
  - Calibration & Model Diagnostics
  - At-Risk Learner Watchlist

## Slide 16. Managerial Implications

- Day 7 can be used for early triage
- Day 30 gives the strongest ranking
- Critical risk band should be prioritized first
- Segments help assign differentiated follow-up

## Slide 17. Limitations and Future Work

- Single dataset
- At-risk target is broad
- Recommendation is not causal
- Future work:
  - temporal sequence modeling
  - external validation
  - intervention impact evaluation

## Slide 18. Conclusion

- The project moved from a static classifier to a research-grade early-warning study
- It links:
  - behavior
  - segmentation
  - recommendation
  - multi-horizon prediction
  - calibration
  - intervention prioritization
