# Experiment design and interpretation

## Data boundary

The Kaggle Store Item Demand Forecasting training file contains daily sales for
500 store-item series from 2013 through 2017. Data are checked for duplicate keys,
missing values, calendar gaps, and negative or nonfinite sales. A missing day is
not silently treated as zero. The file hash and tested package versions are saved
in `results/run_manifest.json`. Raw competition data are not redistributed here.

Sales are used as a proxy for unconstrained demand. The dataset does not identify
stockouts, inventory positions, prices, promotions, or supplier performance.
Consequently, the inventory simulation is a controlled scenario, not a historical
reconstruction of actual store operations.

## Forecast experiment

Four nonoverlapping 28-day holdouts start after 2017-03-31, 2017-06-30,
2017-09-30, and 2017-12-03. At each origin the model is refitted using observations
through that origin only. All predictions for the next 28 days are then made
without observing any sales inside that holdout. These are diagnostic holdouts,
not an untouched final test for hyperparameter selection; no tuning was performed
against them in this implementation.

The global histogram gradient-boosting model uses 7/14/28-day sales lags, day of
week, month, day of month, store ID, and item ID. IDs are numeric splits, not
learned embeddings. The first 28 days of each series are excluded from fitting.
Parameters are fixed at 300 iterations, learning rate 0.08, seed 42. Automatic
early stopping is disabled so it cannot create a random internal validation split.

The seasonal-naive baseline repeats the last observed week. Both predictors
recursively reuse their own predictions when a required lag extends beyond the
origin. The forecasting function accepts observed history and future calendar
dates, not future actual sales. Its recursive behavior is tested explicitly.

WAPE = sum(abs(actual - forecast)) / sum(actual). MAE is the mean absolute unit
error. The headline WAPE pools absolute errors and actual units across holdouts;
it does not average per-SKU percentages. Per-origin and per-series results remain
available to inspect failures hidden by an aggregate score.

The earlier script constructed lags from the complete dataset, while its baseline
used only the final training week. That comparison mixed information sets. Old
headline numbers must not be presented as results from this revised experiment.

## Inventory experiment

This is a separate, daily decision experiment. It is not the same information
schedule as the fixed-origin 28-day forecasting benchmark.

| Stage | Dates | Purpose |
|---|---|---|
| Model fitting | 2013-01-01 to 2016-09-30 | Fit model and historical-mean baseline |
| Buffer calibration | 2016-10-01 to 2016-12-31 | Estimate pre-test protection-period errors |
| Simulation | 2017-01-01 to 2017-12-31 | Run daily stock movements |
| Scored period | 2017-01-29 to 2017-12-31 | Exclude 28 initialization days |

The fitted model is frozen during calibration and test. Forecast inputs update
daily with observed demand through yesterday; no future actual demand is passed
to a decision. Historical mean, seasonal naive, and gradient boosting use the
same history cutoff, simulator, lead time, calibration rule, and cost assumptions.

### Buffers and daily decisions

- Fixed replenishment lead time: 7 days; review interval: 1 day.
- Forecast protection interval: 8 days (lead time plus review interval).
- For each policy and SKU, calculate actual minus forecast 8-day demand during
  calibration. Keep only windows ending by 2016-12-31: 85 overlapping windows.
- The buffer is the nonnegative 95th percentile of those errors, rounded upward.
  This quantile is a calibration setting, not a guarantee of 95% cycle service or
  95% fill rate. Calibration windows overlap and are not independent samples.
- At the start of each test day, receive due orders, then order up to the ceiling
  of forecast 8-day demand plus the calibrated buffer. Subtract both on-hand and
  outstanding orders when calculating the order quantity.
- An order placed on day t arrives before demand on day t+7. Fulfill demand from
  on-hand stock; unfilled demand is lost, not backlogged. Charge holding cost on
  end-of-day stock. Daily mass conservation and nonnegative inventory are checked.

Each policy starts with on-hand inventory equal to its first order-up-to target,
and no outstanding orders. The first 28 days are excluded from scored metrics to
reduce initialization effects. This does not prove complete steady state.

### Metrics and costs

Fill rate is total fulfilled units / total demand units. Stockout SKU-day rate is
the fraction of scored SKU-days with any lost units; it is not cycle service.
Average inventory is scored end-of-day stock averaged over SKUs and days.

Illustrative cost = 0.02 * inventory unit-days + 1.00 * lost units. These are
dimensionless scenario assumptions, not SGD prices, actual margins, or measured
savings. The nine-scenario sensitivity table varies holding cost over
0.01/0.02/0.05 and lost-unit penalty over 0.5/1.0/2.0, holding policy decisions fixed.
This is cost sensitivity, not an optimization or a policy retuning exercise.

The simulation assumes unconstrained supply, fixed lead time, no ordering or
purchase costs, no capacity or minimum-order constraints, no shelf-life limits,
and full observation of demand even after a simulated lost sale. Ending inventory
and outstanding orders are reported; no terminal salvage or purchase valuation is
included. Results are retrospective comparisons under these assumptions, not a
causal estimate of deploying a policy.

`safety_stock.py` separately retains a simple normal-demand, seven-day reorder-point
reference fitted through 2016-09-30. That reference is not the empirical-buffer
daily-review policy in the backtest and has not demonstrated 95% achieved service.

## High-sales-day description

Daily item sales are pooled across stores. Days at or above the item's 90th
percentile rank are compared with its full-period median daily sales. Ties can
make the selected fraction differ from exactly 10%. This is an in-sample tail
description: selecting high-sales days already selects larger outcomes. It cannot
identify a promotion, a holiday effect, or incremental sales caused by either.

## References

- [scikit-learn: Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [scikit-learn: Lagged features for time-series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)
- [MIT OpenCourseWare: Periodic-review inventory systems](https://ocw.mit.edu/courses/15-760b-introduction-to-operations-management-spring-2004/a6608e17a8ec8957ab797240629d9d6a_lec6_invntry.pdf)
