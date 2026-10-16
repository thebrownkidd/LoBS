"""Comparison mode: run the same world separately on several backends and compare measures.

Engines are never mixed in one live simulation. Integration, contact and constraint handling differ,
so identical trajectories are not expected; this reports *how much* they differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import math3d as m3
from .errors import LobsError, WorldValidationError
from .trajectory import Trajectory


@dataclass
class Metrics:
    rmse: float
    max_abs: float
    final_abs: float
    n: int


@dataclass
class Comparison:
    backends: list[str]
    reference: str
    trajectories: dict[str, Trajectory] = field(default_factory=dict)
    metrics: dict[str, dict[str, Metrics]] = field(
        default_factory=dict
    )  # measure -> backend -> Metrics
    runtime_per_tick: dict[str, float] = field(default_factory=dict)  # wall-clock seconds
    approximations: dict[str, list[str]] = field(default_factory=dict)
    engine_versions: dict[str, str] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)  # backend -> why it did not run
    capability_differences: dict[str, dict[str, str]] = field(default_factory=dict)

    def table(self) -> str:
        """Human-readable summary of errors against the reference backend."""
        others = [b for b in self.backends if b != self.reference and b in self.trajectories]
        lines = [f"reference: {self.reference} ({self.engine_versions.get(self.reference, '?')})"]
        for b, why in self.errors.items():
            lines.append(f"  {b}: NOT RUN - {why}")
        head = f"{'measure':<24}" + "".join(
            f"{b + ' rmse':>16}{b + ' max':>14}{b + ' final':>14}" for b in others
        )
        lines += [head, "-" * len(head)]
        for name, per in self.metrics.items():
            row = f"{name:<24}"
            for b in others:
                m = per.get(b)
                row += (
                    f"{m.rmse:>16.3e}{m.max_abs:>14.3e}{m.final_abs:>14.3e}"
                    if m
                    else f"{'n/a':>44}"
                )
            lines.append(row)
        lines.append(
            "runtime per tick: "
            + ", ".join(f"{b} {t * 1e3:.3f} ms" for b, t in self.runtime_per_tick.items())
        )
        for b, notes in self.approximations.items():
            for n in notes:
                lines.append(f"approximation [{b}]: {n}")
        for cap, per in self.capability_differences.items():
            lines.append(f"capability {cap}: " + ", ".join(f"{b}={lvl}" for b, lvl in per.items()))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


def _diff(a: np.ndarray, b: np.ndarray, angle: bool) -> np.ndarray:
    d = a - b
    if angle:
        d = m3.wrap_angle(d)
    if d.ndim > 1:
        d = np.linalg.norm(d.reshape(len(d), -1), axis=1)
    return np.abs(d)


def compare(world, backends=("mujoco", "chrono"), *, seconds=None, ticks=None, controller=None,
            measures=None, seed=0) -> Comparison:  # fmt: skip
    """Run ``world`` on each backend and compare ``measures`` (default: all) against the first one."""
    backends = list(backends)
    if len(backends) < 2:
        raise LobsError("compare needs at least two backends")
    comp = Comparison(backends=backends, reference=backends[0])
    cap_levels: dict[str, dict[str, str]] = {}
    for name in backends:
        try:
            sim = world.simulate(name, seed=seed)
        except (WorldValidationError, LobsError) as e:
            comp.errors[name] = str(e).replace("\n", " ")
            continue
        t0 = time.perf_counter()
        sim.run(seconds=seconds, ticks=ticks, controller=controller)
        wall = time.perf_counter() - t0
        comp.runtime_per_tick[name] = wall / max(sim.tick, 1)
        comp.trajectories[name] = sim.trajectory(measures)
        comp.approximations[name] = list(sim.info.get("approximations", []))
        comp.engine_versions[name] = str(sim.info.get("version", "?"))
        for cap, sup in sim.capabilities.items():
            cap_levels.setdefault(cap.value, {})[name] = sup.level
        sim.close()
    comp.capability_differences = {k: v for k, v in cap_levels.items() if len(set(v.values())) > 1}
    ref = comp.trajectories.get(comp.reference)
    if ref is None:
        return comp
    for name, ref_values in ref.measures.items():
        meta = ref.meta["measures"].get(name, {})
        if meta.get("kind") == "quat":
            continue
        angle = meta.get("unit", "") == "radian"
        for other in backends[1:]:
            tr = comp.trajectories.get(other)
            if tr is None or name not in tr.measures:
                continue
            n = min(len(ref_values), len(tr.measures[name]))
            a = ref_values[:n].astype(float)
            b = tr.measures[name][:n].astype(float)
            d = _diff(a, b, angle)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            d = d[ok]
            comp.metrics.setdefault(name, {})[other] = Metrics(
                rmse=float(np.sqrt(np.mean(d**2))),
                max_abs=float(d.max()),
                final_abs=float(d[-1]),
                n=int(ok.sum()),
            )
    return comp
