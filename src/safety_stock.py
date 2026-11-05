"""Static normal-demand reference fitted before inventory calibration and test.

This reference is not the empirical-buffer policy used in the inventory backtest.
95% is an assumed cycle-service target, not an observed fill rate.
"""
import argparse
import numpy as np
import pandas as pd
from .common import DEFAULT_DATA, DEFAULT_OUT, load_data, output_dir


def run(data_path=DEFAULT_DATA, out_path=DEFAULT_OUT, cutoff="2016-09-30"):
    frame = load_data(data_path)
    stats = frame[frame.date <= pd.Timestamp(cutoff)].groupby(["store", "item"]).sales.agg(["mean", "std"])
    stats["safety_stock"] = np.ceil(1.645 * stats["std"] * np.sqrt(7)).astype(int)
    stats["reorder_point"] = np.ceil(stats["mean"] * 7 + 1.645 * stats["std"] * np.sqrt(7)).astype(int)
    stats = stats.rename(columns={"mean": "mean_daily", "std": "std_daily"}).reset_index()
    stats["training_cutoff"] = cutoff
    stats.sort_values("mean_daily", ascending=False).head(20).to_csv(
        output_dir(out_path) / "safety_stock_top20.csv", index=False, float_format="%.4f")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args(); run(args.data, args.out)


if __name__ == "__main__":
    main()
