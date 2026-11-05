"""Fixed-origin recursive forecasting with identical information for all models."""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from .common import DEFAULT_DATA, DEFAULT_OUT, Panel, error_metrics, load_data, output_dir, plot_style

FEATURES = ["lag_7", "lag_14", "lag_28", "dow", "month", "day", "store", "item"]
ORIGINS = ["2017-03-31", "2017-06-30", "2017-09-30", "2017-12-03"]
MODEL_PARAMS = {"max_iter": 300, "learning_rate": .08, "random_state": 42, "early_stopping": False}
LABELS = {"seasonal_naive": "Seasonal naive", "gradient_boosting": "Gradient boosting",
          "historical_mean": "Historical mean"}


def add_features(frame):
    frame = frame.sort_values(["store", "item", "date"]).copy()
    group = frame.groupby(["store", "item"]).sales
    for lag in (7, 14, 28):
        frame["lag_%s" % lag] = group.shift(lag)
    frame["dow"] = frame.date.dt.dayofweek
    frame["month"] = frame.date.dt.month
    frame["day"] = frame.date.dt.day
    return frame


def fit_model(observed):
    """Caller supplies observed history only, never future evaluation rows."""
    train = add_features(observed).dropna(subset=["lag_28"])
    if train.empty:
        raise ValueError("At least 29 observed days per series are required.")
    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(train[FEATURES], train.sales)
    return model


def forecast_horizon(past_sales, future_dates, pairs, policy, model=None, means=None):
    """History ends BEFORE future_dates[0]. No future actual values are accepted."""
    past = np.asarray(past_sales, dtype=float)
    dates = pd.DatetimeIndex(future_dates)
    if past.ndim != 2 or len(past) < 28 or past.shape[1] != len(pairs):
        raise ValueError("A 28-day history is required for every series.")
    if not np.isfinite(past).all() or (past < 0).any():
        raise ValueError("History must be finite and nonnegative.")
    if len(dates) == 0 or not dates.equals(pd.date_range(dates[0], periods=len(dates))):
        raise ValueError("Forecast dates must be consecutive daily dates.")
    window = list(past[-28:].copy())
    predictions = []
    for day in dates:
        if policy == "seasonal_naive":
            pred = window[-7].copy()
        elif policy == "historical_mean":
            if means is None:
                raise ValueError("Supply historical means fitted before forecasting.")
            pred = np.asarray(means, dtype=float).copy()
        elif policy == "gradient_boosting":
            if model is None:
                raise ValueError("Supply a fitted model.")
            x = pd.DataFrame({
                "lag_7": window[-7], "lag_14": window[-14], "lag_28": window[-28],
                "dow": day.dayofweek, "month": day.month, "day": day.day,
                "store": pairs.get_level_values("store"),
                "item": pairs.get_level_values("item")})
            pred = np.maximum(model.predict(x[FEATURES]), 0)
        else:
            raise ValueError("Unknown policy: " + policy)
        if pred.shape != (len(pairs),) or not np.isfinite(pred).all() or (pred < 0).any():
            raise ValueError("Forecasts must be finite and nonnegative for every series.")
        window.append(pred)
        predictions.append(pred)
    return np.asarray(predictions)


def predict_origin(frame, panel, origin, horizon=28):
    """Create a frozen-origin experiment; future outcomes are returned only for scoring."""
    cut = pd.Timestamp(origin)
    pos = panel.index(cut) + 1
    if pos + horizon > len(panel.dates):
        raise ValueError("The full holdout must exist: " + str(origin))
    dates = panel.dates[pos:pos + horizon]
    actual = panel.sales[pos:pos + horizon]
    model = fit_model(frame[frame.date <= cut])
    predictions = {
        policy: forecast_horizon(panel.sales[:pos], dates, panel.pairs, policy, model)
        for policy in ("seasonal_naive", "gradient_boosting")
    }
    return dates, actual, predictions


def run(data_path=DEFAULT_DATA, out_path=DEFAULT_OUT, origins=None):
    frame = load_data(data_path)
    panel = Panel.from_frame(frame)
    out = output_dir(out_path)
    origins = ORIGINS if origins is None else origins
    rows, series_rows, samples = [], [], {}
    for origin in origins:
        dates, actual, predictions = predict_origin(frame, panel, origin)
        for policy in ("seasonal_naive", "gradient_boosting"):
            pred = predictions[policy]
            rows.append({"origin": origin, "test_end": str(dates[-1].date()),
                         "model": policy, **error_metrics(actual, pred)})
            for j, (store, item) in enumerate(panel.pairs):
                series_rows.append({"origin": origin, "store": store, "item": item,
                                    "model": policy, **error_metrics(actual[:, j], pred[:, j])})
            if origin == origins[-1]:
                samples[policy] = pred[:, 0]
        print("Forecast holdout: %s -> %s" % (origin, dates[-1].date()), flush=True)
    by_origin = pd.DataFrame(rows)
    by_series = pd.DataFrame(series_rows)
    totals = by_origin.groupby("model")[["absolute_error_units", "actual_units", "observations"]].sum()
    totals["MAE"] = totals.absolute_error_units / totals.observations
    totals["WAPE_pct"] = 100 * totals.absolute_error_units / totals.actual_units
    summary = totals.reset_index()[["model", "MAE", "WAPE_pct", "observations"]]
    by_origin.to_csv(out / "forecast_by_origin.csv", index=False, float_format="%.6f")
    by_series.to_csv(out / "forecast_by_series.csv", index=False, float_format="%.6f")
    summary.to_csv(out / "forecast_metrics.csv", index=False, float_format="%.6f")
    plt = plot_style()
    table = by_origin.pivot(index="origin", columns="model", values="WAPE_pct").reindex(origins)
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(table))
    for k, policy in enumerate(("seasonal_naive", "gradient_boosting")):
        bars = ax.bar(x + (k - .5) * .35, table[policy], .35,
                      label=LABELS[policy], color=["#9aacc2", "#167c80"][k])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(x, table.index)
    ax.set(ylabel="WAPE (%) - lower is better", xlabel="Forecast origin (last observed day)",
           title="28-day fixed-origin forecasts | identical information")
    ax.set_ylim(0, table.to_numpy().max() * 1.28)
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "forecast_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    last = panel.index(origins[-1]) + 1
    start = max(0, last - 62)
    ax.plot(panel.dates[start:last + 28], panel.sales[start:last + 28, 0],
            color="#667085", label="Actual daily sales", linewidth=1.2)
    for policy, color in [("seasonal_naive", "#bd7b34"), ("gradient_boosting", "#167c80")]:
        ax.plot(panel.dates[last:last + 28], samples[policy], color=color, label=LABELS[policy])
    ax.axvline(pd.Timestamp(origins[-1]), color="#344054", ls="--", label="Forecast origin")
    store, item = panel.pairs[0]
    ax.set(title="Store %s / Item %s | 28 predictions made at the dashed line" % (store, item),
           ylabel="Units", xlabel="Date")
    ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(out / "forecast_sample.png"); plt.close(fig)
    print(summary.to_string(index=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--origins", nargs="+", default=ORIGINS)
    args = parser.parse_args()
    run(args.data, args.out, args.origins)


if __name__ == "__main__":
    main()
