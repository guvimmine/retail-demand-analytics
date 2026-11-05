"""Reproducible forecasting and inventory experiments."""
import os
import tempfile
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "retail-demand-matplotlib"))
