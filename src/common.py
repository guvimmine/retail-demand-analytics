"""Validated data, panel representation, metrics, and plotting."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "train.csv"
DEFAULT_OUT = ROOT / "results"


def load_data(path=DEFAULT_DATA):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Download Kaggle train.csv to data/train.csv, or use --data.")
    frame = pd.read_csv(path, parse_dates=["date"])
    required = ["date", "store", "item", "sales"]
    if not set(required).issubset(frame.columns):
        raise ValueError("Required columns: date, store, item, sales.")
    frame = frame[required].copy()
    if frame.isna().any().any():
        raise ValueError("Resolve missing data before running experiments.")
    if not np.isfinite(frame.sales).all() or (frame.sales < 0).any():
        raise ValueError("Sales must be finite and nonnegative.")
    if frame.duplicated(["date", "store", "item"]).any():
        raise ValueError("Duplicate store-item-date keys.")
    return frame.sort_values(["store", "item", "date"]).reset_index(drop=True)


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    pairs: pd.MultiIndex
    sales: np.ndarray

    @classmethod
    def from_frame(cls, frame):
        wide = frame.pivot(index="date", columns=["store", "item"], values="sales")
        wide = wide.sort_index().sort_index(axis=1)
        wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max()))
        if wide.isna().any().any():
            raise ValueError("A complete daily panel is required; missing days are not zeros.")
        return cls(pd.DatetimeIndex(wide.index), wide.columns, wide.to_numpy(float))

    def index(self, date):
        return int(self.dates.get_loc(pd.Timestamp(date)))


def error_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted shapes must match.")
    error = np.abs(actual - predicted)
    total = float(actual.sum())
    if total <= 0:
        raise ValueError("WAPE is undefined when total actual sales are zero.")
    return {"MAE": float(error.mean()), "WAPE_pct": float(error.sum() / total * 100),
            "absolute_error_units": float(error.sum()), "actual_units": total,
            "observations": int(actual.size)}


def output_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_manifest(frame, data_path, out, settings):
    import matplotlib
    import sklearn
    report = {
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "rows": len(frame), "series": len(frame[["store", "item"]].drop_duplicates()),
        "date_start": str(frame.date.min().date()), "date_end": str(frame.date.max().date()),
        "python": platform.python_version(),
        "packages": {"numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "settings": settings,
    }
    (Path(out) / "run_manifest.json").write_text(json.dumps(report, indent=2) + "\n")


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": .18})
    return plt
