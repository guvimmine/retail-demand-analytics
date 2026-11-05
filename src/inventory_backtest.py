"""Daily periodic-review, lost-sales simulation with pre-test error calibration."""
import argparse
import json
import numpy as np
import pandas as pd
from .common import DEFAULT_DATA, DEFAULT_OUT, Panel, load_data, output_dir, plot_style
from .forecast import LABELS, fit_model, forecast_horizon

POLICIES = ("historical_mean", "seasonal_naive", "gradient_boosting")
SETTINGS = {
    "train_end": "2016-09-30", "calibration_start": "2016-10-01",
    "calibration_end": "2016-12-31", "test_start": "2017-01-01",
    "test_end": "2017-12-31", "lead_time_days": 7, "review_days": 1,
    "warmup_days": 28, "error_quantile": .95,
    "holding_cost_per_unit_day": .02, "lost_sales_penalty_per_unit": 1.0,
}


def simulate(demand, targets, dates, lead_time=7, warmup=28, holding_cost=.02, lost_penalty=1.):
    """Receive -> review/order -> demand -> holding cost, once per day.

    Orders placed on day t arrive BEFORE demand on t + lead_time. Unfilled
    demand is lost, never backlogged. Position includes on-hand AND on-order.
    Targets cover lead_time + one review day. Initialization is excluded by warmup.
    """
    demand, targets = np.asarray(demand, float), np.asarray(targets, float)
    if demand.ndim != 2 or targets.shape != demand.shape or len(dates) != len(demand):
        raise ValueError("Demand, targets, and daily dates must align.")
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0 or not dates.equals(pd.date_range(dates[0], periods=len(dates))):
        raise ValueError("Simulation dates must be consecutive daily dates.")
    if not np.isfinite(demand).all() or not np.isfinite(targets).all():
        raise ValueError("Demand and targets must be finite.")
    if (demand < 0).any() or (targets < 0).any():
        raise ValueError("Demand and targets must be nonnegative.")
    if lead_time < 1 or not 0 <= warmup < len(demand):
        raise ValueError("Positive lead time and a nonempty scored period are required.")
    if holding_cost < 0 or lost_penalty < 0:
        raise ValueError("Costs must be nonnegative.")
    days, count = demand.shape
    initial = np.ceil(targets[0]).copy()
    on_hand, on_order = initial.copy(), np.zeros(count)
    pipeline = np.zeros((days + lead_time, count))
    delivered, fulfilled_all = np.zeros(count), np.zeros(count)
    demand_scored, lost_scored, inventory_days, stockout_days = [np.zeros(count) for _ in range(4)]
    records = []
    for t, date in enumerate(dates):
        arrivals = pipeline[t]
        delivered += arrivals
        on_hand += arrivals
        on_order -= arrivals
        quantity = np.maximum(np.ceil(targets[t]) - on_hand - on_order, 0)
        pipeline[t + lead_time] += quantity
        on_order += quantity
        fulfilled = np.minimum(on_hand, demand[t])
        lost = demand[t] - fulfilled
        on_hand -= fulfilled
        fulfilled_all += fulfilled
        # Conservation and non-negativity catch event timing / pipeline mistakes.
        if not np.allclose(initial + delivered - fulfilled_all, on_hand):
            raise AssertionError("Inventory mass balance failed.")
        if (on_hand < -1e-8).any() or (on_order < -1e-8).any():
            raise AssertionError("Negative inventory state.")
        scored = t >= warmup
        if scored:
            demand_scored += demand[t]
            lost_scored += lost
            inventory_days += on_hand
            stockout_days += lost > 0
        records.append({
            "date": str(pd.Timestamp(date).date()), "scored": scored,
            "demand_units": demand[t].sum(), "fulfilled_units": fulfilled.sum(),
            "lost_units": lost.sum(), "end_inventory_units": on_hand.sum(),
            "on_order_units": on_order.sum(), "arrivals_units": arrivals.sum(),
            "ordered_units": quantity.sum(), "order_lines": int((quantity > 0).sum()),
            "stockout_series": int((lost > 0).sum()),
        })
    daily = pd.DataFrame(records)
    scored_daily = daily[daily.scored]
    total = demand_scored.sum()
    row = {
        "scored_start": scored_daily.date.iloc[0], "scored_end": scored_daily.date.iloc[-1],
        "scored_days": len(scored_daily), "series": count, "demand_units": total,
        "fulfilled_units": total - lost_scored.sum(), "lost_units": lost_scored.sum(),
        "fill_rate_pct": 100 * (1 - lost_scored.sum() / total) if total else 100.,
        "stockout_sku_day_pct": 100 * stockout_days.sum() / (len(scored_daily) * count),
        "avg_end_inventory_per_sku": inventory_days.sum() / (len(scored_daily) * count),
        "inventory_unit_days": inventory_days.sum(),
        "holding_cost": holding_cost * inventory_days.sum(),
        "lost_sales_cost": lost_penalty * lost_scored.sum(),
        "scored_order_lines": int(scored_daily.order_lines.sum()),
        "final_on_hand_units": float(on_hand.sum()), "final_on_order_units": float(on_order.sum()),
    }
    row["illustrative_total_cost"] = row["holding_cost"] + row["lost_sales_cost"]
    per_series = pd.DataFrame({
        "demand_units": demand_scored, "lost_units": lost_scored,
        "fill_rate_pct": 100 * np.divide(demand_scored - lost_scored, demand_scored,
                                        out=np.ones(count), where=demand_scored > 0),
        "stockout_days": stockout_days.astype(int),
        "avg_end_inventory_units": inventory_days / len(scored_daily),
    })
    return row, daily, per_series


def calibrated_buffer(actual_totals, forecast_totals, quantile):
    actual, predicted = np.asarray(actual_totals), np.asarray(forecast_totals)
    if actual.shape != predicted.shape or actual.ndim != 2 or len(actual) < 2:
        raise ValueError("At least two aligned calibration windows are required.")
    if not 0 < quantile < 1:
        raise ValueError("Error quantile must be strictly between zero and one.")
    return np.ceil(np.maximum(np.quantile(actual - predicted, quantile, axis=0), 0))


def run(data_path=DEFAULT_DATA, out_path=DEFAULT_OUT, settings=None):
    cfg = dict(SETTINGS)
    if settings:
        cfg.update(settings)
    frame, out = load_data(data_path), output_dir(out_path)
    panel = Panel.from_frame(frame)
    train_end = pd.Timestamp(cfg["train_end"])
    cal_start, cal_end = pd.Timestamp(cfg["calibration_start"]), pd.Timestamp(cfg["calibration_end"])
    test_start, test_end = pd.Timestamp(cfg["test_start"]), pd.Timestamp(cfg["test_end"])
    if not train_end < cal_start <= cal_end < test_start <= test_end:
        raise ValueError("Training, calibration, and test must be ordered and disjoint.")
    if cfg["review_days"] != 1:
        raise ValueError("This implementation supports daily review only.")
    horizon = cfg["lead_time_days"] + cfg["review_days"]
    start_i, end_i = panel.index(cal_start), panel.index(test_end) + 1
    test_i, cal_end_i = panel.index(test_start), panel.index(cal_end) + 1
    train_i = panel.index(train_end) + 1
    model = fit_model(frame[frame.date <= train_end])
    means = panel.sales[:train_i].mean(axis=0)
    forecasts = {p: [] for p in POLICIES}
    for i in range(start_i, end_i):
        dates = pd.date_range(panel.dates[i], periods=horizon)
        # Only history through yesterday is passed to the predictor.
        past = panel.sales[:i]
        for policy in POLICIES:
            pred = forecast_horizon(past, dates, panel.pairs, policy, model, means)
            forecasts[policy].append(pred.sum(axis=0))
        if (i - start_i + 1) % 90 == 0:
            print("Inventory forecast origins processed: %s" % (i - start_i + 1), flush=True)
    forecasts = {p: np.asarray(v) for p, v in forecasts.items()}
    cal_indices = list(range(start_i, cal_end_i - horizon + 1))
    actual_cal = np.asarray([panel.sales[i:i + horizon].sum(axis=0) for i in cal_indices])
    test_dates, test_demand = panel.dates[test_i:end_i], panel.sales[test_i:end_i]
    metric_rows, daily_tables, series_tables, buffers, sensitivity = [], [], [], [], []
    for policy in POLICIES:
        buffer = calibrated_buffer(actual_cal, forecasts[policy][:len(cal_indices)], cfg["error_quantile"])
        target = forecasts[policy][test_i - start_i:] + buffer
        row, daily, series = simulate(
            test_demand, target, test_dates, cfg["lead_time_days"], cfg["warmup_days"],
            cfg["holding_cost_per_unit_day"], cfg["lost_sales_penalty_per_unit"])
        metric_rows.append({"policy": policy, **row})
        daily.insert(0, "policy", policy); daily_tables.append(daily)
        series.insert(0, "item", panel.pairs.get_level_values("item"))
        series.insert(0, "store", panel.pairs.get_level_values("store"))
        series.insert(0, "policy", policy); series_tables.append(series)
        buffers.append(pd.DataFrame({"policy": policy,
            "store": panel.pairs.get_level_values("store"), "item": panel.pairs.get_level_values("item"),
            "buffer_units": buffer, "calibration_windows": len(cal_indices)}))
        for holding in (.01, .02, .05):
            for penalty in (.5, 1., 2.):
                sensitivity.append({"policy": policy, "holding_cost_per_unit_day": holding,
                    "lost_sales_penalty_per_unit": penalty,
                    "illustrative_total_cost": holding * row["inventory_unit_days"] + penalty * row["lost_units"]})
    metrics = pd.DataFrame(metric_rows)
    daily = pd.concat(daily_tables, ignore_index=True)
    metrics.to_csv(out / "inventory_metrics.csv", index=False, float_format="%.6f")
    daily.to_csv(out / "inventory_daily.csv", index=False, float_format="%.6f")
    pd.concat(series_tables, ignore_index=True).to_csv(out / "inventory_by_series.csv", index=False, float_format="%.6f")
    pd.concat(buffers, ignore_index=True).to_csv(out / "inventory_buffers.csv", index=False, float_format="%.6f")
    pd.DataFrame(sensitivity).to_csv(out / "inventory_cost_sensitivity.csv", index=False, float_format="%.6f")
    cfg.update({"protection_horizon_days": horizon, "calibration_windows_per_series": len(cal_indices),
                "cost_units": "illustrative, not currency or measured business savings"})
    (out / "inventory_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    plot_results(metrics, daily, out)
    print(metrics[["policy", "fill_rate_pct", "avg_end_inventory_per_sku",
                   "stockout_sku_day_pct", "illustrative_total_cost"]].to_string(index=False), flush=True)
    return metrics


def plot_results(metrics, daily, out):
    plt = plot_style()
    colors = ["#9aacc2", "#bd7b34", "#167c80"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        ("fill_rate_pct", "Units fulfilled (%) - higher is better"),
        ("avg_end_inventory_per_sku", "Average end-of-day inventory / SKU (units)"),
        ("stockout_sku_day_pct", "SKU-days with lost sales (%) - lower is better"),
        ("illustrative_total_cost", "Illustrative cost units - not measured savings"),
    ]
    for ax, (field, title) in zip(axes.flat, panels):
        bars = ax.bar(range(len(metrics)), metrics[field], color=colors)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_xticks(range(len(metrics)), ["Historical\nmean", "Seasonal\nnaive", "Gradient\nboosting"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(float(metrics[field].max()) * 1.2, 1))
    first = metrics.iloc[0]
    fig.suptitle("Inventory backtest | %s to %s | %s series" %
                 (first.scored_start, first.scored_end, int(first.series)), fontsize=13)
    fig.tight_layout(); fig.savefig(out / "inventory_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for policy, color in zip(POLICIES, colors):
        subset = daily[(daily.policy == policy) & daily.scored].copy()
        ax.plot(pd.to_datetime(subset.date),
                (subset.end_inventory_units / int(metrics.series.iloc[0])).rolling(7).mean(),
                label=LABELS[policy], color=color)
    ax.set(title="Average inventory per SKU | 7-day trailing mean",
           ylabel="End-of-day units / SKU", xlabel="Date")
    ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out / "inventory_timeline.png"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(); run(args.data, args.out)


if __name__ == "__main__":
    main()
