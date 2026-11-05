"""Behavioral checks for information availability and inventory event timing."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.common import Panel, error_metrics
from src.forecast import forecast_horizon, predict_origin
from src.inventory_backtest import calibrated_buffer, simulate


class LagSevenPlusOne:
    def predict(self, features):
        return features.lag_7.to_numpy() + 1


class ForecastBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.MultiIndex.from_tuples([(1, 1)], names=["store", "item"])
        self.past = np.arange(1, 29, dtype=float).reshape(-1, 1)
        self.dates = pd.date_range("2020-01-29", periods=28)

    def test_weekly_baseline_replays_observed_week_for_entire_horizon(self):
        actual = forecast_horizon(self.past, self.dates, self.pairs, "seasonal_naive")
        np.testing.assert_array_equal(actual[:, 0], np.tile(np.arange(22, 29), 4))

    def test_model_recursively_uses_predictions_after_week_one(self):
        actual = forecast_horizon(self.past, self.dates, self.pairs,
                                  "gradient_boosting", LagSevenPlusOne())
        for week in range(4):
            np.testing.assert_array_equal(actual[week * 7:(week + 1) * 7, 0],
                                          np.arange(22, 29) + week + 1)

    def test_input_history_is_not_overwritten_by_predictions(self):
        before = self.past.copy()
        forecast_horizon(self.past, self.dates, self.pairs, "gradient_boosting", LagSevenPlusOne())
        np.testing.assert_array_equal(before, self.past)

    def test_missing_calendar_days_are_rejected_instead_of_shifting_wrong_lags(self):
        frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-03"]),
                              "store": [1, 1], "item": [1, 1], "sales": [10, 20]})
        with self.assertRaisesRegex(ValueError, "complete daily panel"):
            Panel.from_frame(frame)

    def test_metrics_pool_units_instead_of_averaging_percentage_errors(self):
        metric = error_metrics(np.array([1, 100]), np.array([0, 90]))
        self.assertAlmostEqual(metric["WAPE_pct"], 11 / 101 * 100)

    def test_changing_all_holdout_actuals_cannot_change_either_forecast(self):
        dates = pd.date_range("2020-01-01", periods=90)
        origin = dates[59]
        frame = pd.DataFrame({"date": dates, "store": 1, "item": 1,
                              "sales": np.arange(1, 91, dtype=float)})
        changed = frame.copy()
        changed.loc[changed.date > origin, "sales"] += 99999
        fits = []

        def fake_fit(observed):
            self.assertLessEqual(observed.date.max(), origin)
            fits.append(observed.copy())
            return LagSevenPlusOne()

        with patch("src.forecast.fit_model", side_effect=fake_fit):
            _, actual_a, pred_a = predict_origin(frame, Panel.from_frame(frame), origin)
            _, actual_b, pred_b = predict_origin(changed, Panel.from_frame(changed), origin)
        pd.testing.assert_frame_equal(fits[0], fits[1])
        self.assertFalse(np.array_equal(actual_a, actual_b))
        for policy in pred_a:
            np.testing.assert_array_equal(pred_a[policy], pred_b[policy])


class InventoryEventTests(unittest.TestCase):
    def setUp(self):
        self.demand = np.full((5, 1), 8.)
        self.target = np.full((5, 1), 20.)
        self.dates = pd.date_range("2020-01-01", periods=5)

    def test_orders_arrive_only_after_lead_time_and_on_order_is_counted(self):
        row, daily, _ = simulate(self.demand, self.target, self.dates, lead_time=2, warmup=0)
        np.testing.assert_array_equal(daily.arrivals_units, [0, 0, 0, 8, 8])
        np.testing.assert_array_equal(daily.ordered_units, [0, 8, 8, 4, 8])
        np.testing.assert_array_equal(daily.end_inventory_units, [12, 4, 0, 0, 0])
        self.assertEqual(row["lost_units"], 4)
        self.assertEqual(row["fill_rate_pct"], 90)
        self.assertEqual(row["final_on_order_units"], 12)
        self.assertAlmostEqual(row["illustrative_total_cost"], 4.32)

    def test_warmup_losses_and_inventory_are_excluded_from_scored_metrics(self):
        row, daily, _ = simulate(self.demand, self.target, self.dates, lead_time=2, warmup=3)
        self.assertEqual(row["scored_days"], 2)
        self.assertEqual(row["demand_units"], 16)
        self.assertEqual(row["lost_units"], 0)
        self.assertEqual(row["fill_rate_pct"], 100)
        self.assertEqual(int(daily.scored.sum()), 2)

    def test_zero_demand_has_no_lost_sales_or_duplicate_orders(self):
        row, daily, _ = simulate(np.zeros((5, 1)), np.full((5, 1), 5.),
                                 self.dates, lead_time=2, warmup=0)
        self.assertEqual(row["lost_units"], 0)
        self.assertEqual(row["fill_rate_pct"], 100)
        self.assertEqual(daily.ordered_units.sum(), 0)

    def test_higher_safety_quantile_never_reduces_buffer(self):
        actual = np.array([[10., 1], [20, 2], [30, 3], [40, 4]])
        predicted = np.full_like(actual, 15.)
        low = calibrated_buffer(actual, predicted, .5)
        high = calibrated_buffer(actual, predicted, .95)
        self.assertTrue((high >= low).all())
        self.assertTrue((low >= 0).all())

    def test_negative_demand_and_invalid_lead_time_are_rejected(self):
        with self.assertRaises(ValueError):
            simulate(-self.demand, self.target, self.dates, lead_time=2, warmup=0)
        with self.assertRaises(ValueError):
            simulate(self.demand, self.target, self.dates, lead_time=0, warmup=0)

    def test_calendar_gaps_cannot_shorten_the_physical_lead_time(self):
        dates = self.dates.to_list()
        dates[-1] += pd.Timedelta(days=1)
        with self.assertRaisesRegex(ValueError, "consecutive"):
            simulate(self.demand, self.target, dates, lead_time=2, warmup=0)


if __name__ == "__main__":
    unittest.main()
