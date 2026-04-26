from __future__ import annotations

import unittest

import pandas as pd

from src.models.multi_horizon_early_warning import _choose_operating_point, _select_champion


class ModelSelectionRuleTests(unittest.TestCase):
    def test_choose_operating_point_prioritizes_precision_after_recall_floor(self) -> None:
        rows = [
            {"threshold": 0.20, "precision": 0.55, "recall": 0.95, "f2": 0.82, "pr_auc": 0.80},
            {"threshold": 0.25, "precision": 0.60, "recall": 0.91, "f2": 0.81, "pr_auc": 0.81},
            {"threshold": 0.30, "precision": 0.70, "recall": 0.89, "f2": 0.80, "pr_auc": 0.83},
        ]

        selected = _choose_operating_point(rows, target_recall=0.90)
        self.assertEqual(selected["threshold"], 0.25)

    def test_choose_operating_point_breaks_precision_tie_with_f2_then_pr_auc(self) -> None:
        rows = [
            {"threshold": 0.20, "precision": 0.60, "recall": 0.92, "f2": 0.83, "pr_auc": 0.81},
            {"threshold": 0.25, "precision": 0.60, "recall": 0.93, "f2": 0.84, "pr_auc": 0.79},
            {"threshold": 0.30, "precision": 0.60, "recall": 0.93, "f2": 0.84, "pr_auc": 0.82},
        ]

        selected = _choose_operating_point(rows, target_recall=0.90)
        self.assertEqual(selected["threshold"], 0.30)

    def test_select_champion_prefers_earlier_horizon_when_scores_are_close(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "horizon_day": 7,
                    "model": "Logistic Regression",
                    "recall": 0.93,
                    "pr_auc": 0.815,
                    "f2": 0.840,
                    "precision": 0.58,
                },
                {
                    "horizon_day": 14,
                    "model": "XGBoost",
                    "recall": 0.93,
                    "pr_auc": 0.820,
                    "f2": 0.845,
                    "precision": 0.59,
                },
            ]
        )

        champion = _select_champion(selected, target_recall=0.90)
        self.assertEqual(int(champion["horizon_day"]), 7)

    def test_select_champion_keeps_stronger_later_horizon_when_gap_is_not_close(self) -> None:
        selected = pd.DataFrame(
            [
                {
                    "horizon_day": 7,
                    "model": "Logistic Regression",
                    "recall": 0.93,
                    "pr_auc": 0.79,
                    "f2": 0.83,
                    "precision": 0.57,
                },
                {
                    "horizon_day": 30,
                    "model": "XGBoost",
                    "recall": 0.92,
                    "pr_auc": 0.86,
                    "f2": 0.85,
                    "precision": 0.63,
                },
            ]
        )

        champion = _select_champion(selected, target_recall=0.90)
        self.assertEqual(int(champion["horizon_day"]), 30)
        self.assertEqual(champion["model"], "XGBoost")


if __name__ == "__main__":
    unittest.main()
