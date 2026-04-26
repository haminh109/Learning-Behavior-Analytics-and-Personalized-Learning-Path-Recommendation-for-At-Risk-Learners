# Demo Talk Track

This is a short speaking script for demoing the upgraded project in 5 to 7 minutes.

## 1. Opening

> This project studies learner behavior in OULAD and asks a practical question: how early can we identify learners who are likely to fail or withdraw, and how can we connect that signal to useful learner recommendations?

## 2. Problem Scale

> The dataset contains 32,593 enrollments, and 52.8% of them are at risk under the project's definition of fail or withdrawn. So this is not a marginal problem. It is a large operational problem.

## 3. Behavior Evidence

> The EDA shows that weak outcomes appear early. Withdrawn learners already have extremely weak first-two-week engagement, and assessment completion separates successful learners from weak outcomes much more clearly than demographics alone.

## 4. Segmentation Layer

> Before building the model, I segmented learners into four groups: Inactive Drop-offs, Sporadic Explorers, Steady Progressors, and Focused Achievers. This layer matters because the project is not only about prediction. It is also about explaining who these learners are and what kind of intervention fits them.

## 5. Recommendation Layer

> The recommendation engine compares each learner with successful peers in the same cluster, then assigns a path like Consistency Building, Early Start, or Assessment Recovery. So the project already has an action layer before the predictive model is introduced.

## 6. Multi-Horizon Modeling

> The main upgrade is that the model is no longer trained only at day 30. I built four leakage-safe feature sets at day 7, 14, 21, and 30, then compared Logistic Regression, Random Forest, and XGBoost at each horizon.

## 7. Earliest Useful Horizon

> Day 7 is already useful. The best day-7 operating point reaches validation recall of 93.9%, which means the system can begin triaging risk very early, even though the ranking quality is weaker than later horizons.

## 8. Final Champion

> The final champion is XGBoost at day 30 with a threshold of 0.25. On the held-out test set, recall is 93.7%, precision is 63.1%, F2 is 0.854, ROC-AUC is 0.847, and PR-AUC is 0.877.

## 9. Why This Is Stronger Than a Usual Classifier

> I also added ablation and calibration. Ablation shows that engagement plus assessment features create most of the predictive value, while demographics alone are weak. Calibration and risk bands show that the predicted probabilities are ordered well enough to support intervention tiers like Low, Medium, High, and Critical risk.

## 10. Close

> So the final contribution is not just a classifier. It is a full early-warning research pipeline that explains learner behavior, segments learner types, recommends next actions, compares early-warning horizons, and translates model output into practical risk bands for intervention.
