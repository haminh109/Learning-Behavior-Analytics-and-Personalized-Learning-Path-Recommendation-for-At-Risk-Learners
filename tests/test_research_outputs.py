from __future__ import annotations

import unittest
from pathlib import Path

from src.validation import validate_research_outputs


class ResearchOutputValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.processed_dir = cls.repo_root / "data" / "processed"
        cls.summary = validate_research_outputs(cls.processed_dir)

    def test_horizon_tables_are_clean(self) -> None:
        self.assertTrue((self.summary.horizon_shapes["missing_cells"] == 0).all())
        self.assertTrue((self.summary.horizon_shapes["duplicate_keys"] == 0).all())

    def test_model_comparison_has_all_pairs(self) -> None:
        self.assertEqual(self.summary.model_pair_count, 12)

    def test_ablation_has_expected_row_count(self) -> None:
        self.assertEqual(self.summary.ablation_row_count, 10)

    def test_champion_metrics_include_validation_and_test(self) -> None:
        self.assertEqual(set(self.summary.champion_metrics["split"]), {"validation", "test"})

    def test_risk_band_order_is_monotonic(self) -> None:
        self.assertTrue(self.summary.risk_band_summary["actual_at_risk_rate"].is_monotonic_increasing)
        self.assertTrue(self.summary.risk_band_summary["average_predicted_probability"].is_monotonic_increasing)


if __name__ == "__main__":
    unittest.main()
