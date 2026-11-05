"""Reconcile published results to their detailed evidence without Kaggle data."""
import unittest
from pathlib import Path
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"


class PublishedEvidenceTests(unittest.TestCase):
    def test_forecast_headlines_reconcile_to_all_four_holdouts(self):
        folds = pd.read_csv(RESULTS / "forecast_by_origin.csv")
        summary = pd.read_csv(RESULTS / "forecast_metrics.csv").set_index("model")
        self.assertEqual(folds.origin.nunique(), 4)
        for model, rows in folds.groupby("model"):
            self.assertEqual(rows.observations.sum(), 56000)
            expected = 100 * rows.absolute_error_units.sum() / rows.actual_units.sum()
            self.assertAlmostEqual(summary.loc[model, "WAPE_pct"], expected, places=5)

    def test_inventory_metrics_reconcile_to_scored_daily_movements(self):
        daily = pd.read_csv(RESULTS / "inventory_daily.csv")
        summary = pd.read_csv(RESULTS / "inventory_metrics.csv").set_index("policy")
        for policy, rows in daily.groupby("policy"):
            scored = rows[rows.scored]
            self.assertEqual(len(rows), 365)
            self.assertEqual(len(scored), 337)
            metric = summary.loc[policy]
            self.assertEqual(scored.demand_units.sum(), metric.demand_units)
            self.assertEqual(scored.lost_units.sum(), metric.lost_units)
            self.assertAlmostEqual(100 * scored.fulfilled_units.sum() / scored.demand_units.sum(),
                                   metric.fill_rate_pct, places=5)
            self.assertAlmostEqual(scored.end_inventory_units.mean() / metric.series,
                                   metric.avg_end_inventory_per_sku, places=5)
