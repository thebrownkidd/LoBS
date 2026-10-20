"""Turn recorded measure history into plottable traces (no Qt)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

AXES = "xyzw"


def traces(history, name: str, window: float | None = None):
    """``[(label, t, y)]`` for one measure: scalars give one trace, vectors one per component, bools 0/1."""
    t = history.time
    if len(t) == 0:
        return []
    y = np.asarray(history[name], dtype=float)
    lo = 0
    if window is not None:
        lo = int(np.searchsorted(t, t[-1] - window))
    t, y = t[lo:], y[lo:]
    if y.ndim == 1:
        return [(name, t, y)]
    flat = y.reshape(len(y), -1)
    labels = [f"{name}.{AXES[k]}" if flat.shape[1] <= 4 else f"{name}[{k}]" for k in range(flat.shape[1])]
    return [(lab, t, flat[:, k]) for k, lab in enumerate(labels)]


def format_value(v) -> str:
    a = np.asarray(v)
    if a.dtype == bool and a.shape == ():
        return "true" if bool(a) else "false"
    if a.shape == ():
        return f"{float(a):.5g}"
    return "[" + ", ".join(f"{float(x):.4g}" for x in a.ravel()[:6]) + (", ..." if a.size > 6 else "") + "]"


def export_csv(history, names, path) -> Path:
    path = Path(path)
    cols, header = [history.time], ["t"]
    for n in names:
        for lab, _, y in traces(history, n):
            cols.append(y)
            header.append(lab)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in zip(*cols, strict=True):
            w.writerow([f"{float(v):.9g}" for v in row])
    return path
