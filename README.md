# Retail Demand & Inventory Analytics

**Can better demand forecasts support lower inventory without sacrificing availability?**

A reproducible Python case study connecting fixed-origin sales forecasts to a daily
inventory simulation across **500 store-item series**. The emphasis is on fair
evaluation, explicit assumptions, and decisions supported by inspectable results.

## Results at a glance

| Experiment / metric | Seasonal-naive baseline | Gradient boosting |
|---|---:|---:|
| Pooled WAPE over four 28-day holdouts — lower is better | 18.05% | **11.50%** |
| Inventory unit fill rate — higher is better | 98.73% | **99.39%** |
| Average end-of-day inventory per SKU — units | 56.52 | **38.11** |
| SKU-days with any lost demand — lower is better | 5.76% | **4.52%** |

In the scored inventory simulation, gradient boosting held **32.6% less average
inventory** than the seasonal-naive policy while achieving a higher unit fill rate.
These are retrospective scenario results, not measured store savings or a guarantee
of future performance. The forecasting and inventory experiments have different,
explicitly documented decision schedules.

## Data

[Kaggle — Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only):
913,000 daily records, 10 stores × 50 items, 2013-01-01 through 2017-12-31.

The file contains only `date`, `store`, `item`, and `sales`. It has no observed
inventory, supplier lead times, promotion flags, or prices. Sales are treated as a
proxy for unconstrained demand; raw competition data are not redistributed.

## 1. Forecasting: identical information, multiple time periods

At each origin, fit using history through that date and predict all next 28 days
at once. Both methods recursively use their own predictions when an observed lag
is no longer available. **No holdout actual sales enter either forecast.**

| Last observed day | Holdout ends | Seasonal-naive WAPE | Gradient-boosting WAPE |
|---|---|---:|---:|
| 2017-03-31 | 2017-04-28 | 18.13% | 11.48% |
| 2017-06-30 | 2017-07-28 | 13.35% | 10.52% |
| 2017-09-30 | 2017-10-28 | 15.77% | 11.49% |
| 2017-12-03 | 2017-12-31 | 28.91% | 13.20% |

![Four fixed-origin forecast evaluations](results/forecast_comparison.png)

The pooled WAPE of 11.50% uses the sum of absolute errors divided by the sum of
actual units across all four holdouts (56,000 observations). It is not the mean of
per-SKU percentage errors. The pooled relative error reduction is **36.3%** versus
this baseline; individual periods differ substantially.

The global model is scikit-learn's `HistGradientBoostingRegressor`, using 7/14/28-day
lags, calendar features, and store/item IDs. Parameters are fixed, with random
internal early stopping disabled. These are diagnostic holdouts, not an untouched
final test for a tuned model. See [per-origin results](results/forecast_by_origin.csv),
[per-series results](results/forecast_by_series.csv), and [methodology](docs/methodology.md).

![Example fixed-origin forecast for store 1, item 1](results/forecast_sample.png)

## 2. Inventory: forecasts become replenishment decisions

Compare **historical mean**, **seasonal naive**, and **gradient boosting** in the
same daily-review, seven-day-lead-time, lost-sales simulator.

| Stage | Dates | What is allowed |
|---|---|---|
| Fit | Through 2016-09-30 | Fit the model and historical means |
| Calibrate | 2016-10-01 to 2016-12-31 | Estimate policy-specific forecast-error buffers |
| Warm up | 2017-01-01 to 2017-01-28 | Initialize inventory and replenishment pipeline |
| Score | 2017-01-29 to 2017-12-31 | Measure 337 days × 500 series |

Each morning: receive due orders, forecast the next **8 days** (7 lead-time days
plus 1 review day), order up to forecast demand plus a calibrated buffer, then
fulfill that day's demand. Outstanding orders are counted so inventory is not
ordered twice. Model weights and buffers stay frozen during test; input history
updates only with demand observed through yesterday.

Buffers use the nonnegative **95th percentile of pre-test eight-day forecast
errors**, separately for each policy and SKU. This is a calibration parameter:
**it does not mean 95% service was achieved**, and unit fill rate differs from
cycle service level.

| Policy | Unit fill rate | Average inventory / SKU | Stockout SKU-days |
|---|---:|---:|---:|
| Historical mean | 94.07% | 44.08 units | 24.81% |
| Seasonal naive | 98.73% | 56.52 units | 5.76% |
| Gradient boosting | **99.39%** | **38.11 units** | **4.52%** |

![Inventory strategy comparison](results/inventory_comparison.png)
![Inventory through the scored year](results/inventory_timeline.png)

The illustrative cost scenario uses 0.02 per unit of end-of-day stock per day and
1.00 per lost unit, in **scenario cost units, not SGD**. Costs are 319,750.92 for
seasonal naive and 190,989.40 for gradient boosting, a 40.3% scenario reduction.
Gradient boosting has the lowest cost in all nine supplied cost scenarios; the
sensitivity changes cost weights, not the underlying policy decisions.

The simulator assumes unlimited supply, fixed lead time, full observation of
demand, no order/setup costs, no purchase costs, no capacity constraints, and no
shelf-life limits. It is not an optimized or production-ready inventory system.

Evidence: [metrics](results/inventory_metrics.csv),
[daily movements](results/inventory_daily.csv),
[per-SKU outcomes](results/inventory_by_series.csv),
[calibrated buffers](results/inventory_buffers.csv),
[cost sensitivity](results/inventory_cost_sensitivity.csv),
[configuration](results/inventory_config.json).

## 3. High-sales days: description, not promotion effectiveness

The highest approximately 10% of sales days have a median item-level premium of
**46.8%** over the median day. This follows a selection based on sales themselves:
it describes the upper tail, and does **not** identify a promotion or estimate its
causal effect. Ties can change the selected fraction. The former promotion-uplift
naming has been removed.

![Descriptive upper-tail sales distribution](results/high_demand_distribution.png)

The separate [normal-demand safety-stock reference](results/safety_stock_top20.csv)
is fitted through 2016-09-30. It is explanatory and is not the policy evaluated by
the inventory backtest.

## Reproduce

Run from the repository root. Dependencies are pinned; the recorded local run
used Python 3.9.6. Install the pinned versions to reproduce the published results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Download competition train.csv and place it at data/train.csv.
python -m src.run_all
python -m unittest discover -s tests -v
```

To use a different file or output folder:

```bash
python -m src.run_all --data /path/to/train.csv --out /path/to/results
```

Individual modules: `python -m src.forecast`, `python -m src.inventory_backtest`,
`python -m src.safety_stock`, and `python -m src.high_demand`.
They accept `--data` and `--out`; the default dates target this competition dataset.

The pipeline regenerates CSVs, charts, and a
[dataset/environment manifest](results/run_manifest.json). The README snapshot
must be reviewed if experiment settings or data change.

## Validation and limitations

- **14 checks** cover recursive lags, invariance to changed future actuals, calendar
  gaps, pooled metrics, arrival timing, outstanding orders, stock conservation,
  lost sales, warmup exclusion, and reconciliation of published results.
- No Kaggle download is needed for the checks. Full experiments require the data.
- Only one public dataset and one inventory test year are evaluated. Calibration
  windows overlap; no confidence intervals or deployment experiment are claimed.
- The data cannot support promotion attribution or observed financial ROI.
- Actual store stockouts would censor sales; that issue cannot be identified here.

[Full methodology, formulas, and references](docs/methodology.md)
