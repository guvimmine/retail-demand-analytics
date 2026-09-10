# Retail Demand & Inventory Analytics

Personal analytics project: demand forecasting, SKU-level inventory policy, and
high-demand (promotion/holiday) period analysis on a public retail dataset.

**Data**: [Kaggle — Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only)
(`train.csv`, 913k rows: 10 stores × 50 items, daily sales 2013-01-01 → 2017-12-31).
Download it into `data/train.csv` (see Setup).

## What it does

| Script | Output | Headline result |
|---|---|---|
| `src/forecast.py` | `results/forecast_metrics.csv`, `results/forecast_sample.png` | Gradient-boosting forecasts reach **WAPE 13.2% (MAE 5.87)** on a 28-day holdout, **54% lower error** than the seasonal-naive baseline (WAPE 28.9%) |
| `src/safety_stock.py` | `results/safety_stock_top20.csv` | Per-SKU safety stock & reorder points at **95% service level, 7-day lead time** (e.g. top SKU: ROP 809 units) |
| `src/promo_uplift.py` | `results/high_demand_uplift_*.csv`, `results/high_demand_uplift.png` | High-demand periods lift item sales **+46.8% median** vs the median-day baseline (p25–p75: 46.3–47.5%) |

## Method notes

- **Forecasting**: global `HistGradientBoostingRegressor` (scikit-learn) with
  lag features (7/14/28), day-of-week, month, day, store and item IDs.
  Validation = last 28 days. Baseline = seasonal naive (demand 7 days ago,
  replayed weekly).
- **Safety stock**: demand over lead time treated as normal;
  `reorder point = μ_L + z(0.95)·σ_daily·√L`, L = 7 days.
- **High-demand uplift**: per item, the top decile of demand days is treated as
  the high-demand (promotion/holiday) period; uplift is measured against the
  item's median-day baseline. (The dataset carries no promotion flags, so this
  is an association-based proxy, not causal attribution.)

## Setup

```bash
pip install -r requirements.txt
# place the Kaggle train.csv at data/train.csv
python3 src/forecast.py
python3 src/safety_stock.py
python3 src/promo_uplift.py
```
