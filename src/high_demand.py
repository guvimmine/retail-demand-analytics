"""Describe upper-tail sales days; do not estimate promotion effects."""
import argparse
import pandas as pd
from .common import DEFAULT_DATA, DEFAULT_OUT, load_data, output_dir, plot_style


def run(data_path=DEFAULT_DATA, out_path=DEFAULT_OUT):
    frame, out = load_data(data_path), output_dir(out_path)
    daily = frame.groupby(["item", "date"], as_index=False).sales.sum()
    daily["rank"] = daily.groupby("item").sales.rank(pct=True)
    baseline = daily.groupby("item").sales.median().rename("median_daily_sales")
    upper = daily[daily["rank"] >= .9].groupby("item").sales.agg(["mean", "count"])
    result = baseline.to_frame().join(upper).rename(
        columns={"mean": "upper_tail_mean_sales", "count": "upper_tail_days"})
    result["upper_tail_premium_pct"] = 100 * (result.upper_tail_mean_sales / result.median_daily_sales - 1)
    result = result.reset_index()
    summary = pd.DataFrame([{"items": len(result),
        "median_upper_tail_premium_pct": result.upper_tail_premium_pct.median(),
        "p25_pct": result.upper_tail_premium_pct.quantile(.25),
        "p75_pct": result.upper_tail_premium_pct.quantile(.75)}])
    result.to_csv(out / "high_demand_by_item.csv", index=False, float_format="%.4f")
    summary.to_csv(out / "high_demand_summary.csv", index=False, float_format="%.4f")
    plt = plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(result.upper_tail_premium_pct, bins=15, color="#167c80", edgecolor="white")
    ax.set(title="Upper-tail sales days vs each item's median day",
           xlabel="Descriptive premium (%) | NOT promotion effectiveness", ylabel="Number of items")
    fig.tight_layout(); fig.savefig(out / "high_demand_distribution.png"); plt.close(fig)
    print("High-sales-day description only; no promotion or causal attribution.", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(); run(args.data, args.out)


if __name__ == "__main__":
    main()
