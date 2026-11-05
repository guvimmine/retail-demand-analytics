"""Run every experiment and record the exact dataset and environment."""
import argparse
from . import forecast, high_demand, inventory_backtest, safety_stock
from .common import DEFAULT_DATA, DEFAULT_OUT, load_data, output_dir, save_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = output_dir(args.out)
    forecast.run(args.data, out)
    inventory_backtest.run(args.data, out)
    safety_stock.run(args.data, out)
    high_demand.run(args.data, out)
    save_manifest(load_data(args.data), args.data, out, {
        "forecast_origins": forecast.ORIGINS, "forecast_horizon_days": 28,
        "model": forecast.MODEL_PARAMS, "inventory": inventory_backtest.SETTINGS})
    print("Finished. Reproducibility manifest: " + str(out / "run_manifest.json"), flush=True)


if __name__ == "__main__":
    main()
