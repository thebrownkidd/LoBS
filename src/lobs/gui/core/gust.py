"""One-click budgeted disturbances ("inject a gust") built on the library's ``sim.apply_force``."""

from __future__ import annotations

import math

import numpy as np

from ...errors import LobsError


def gust_plan(peak: float, impulse: float, tick_dt: float) -> tuple[int, float]:
    """``(ticks, magnitude)`` so that ``magnitude <= peak`` and ``magnitude * ticks * tick_dt == impulse``."""
    if peak <= 0 or impulse <= 0 or tick_dt <= 0:
        raise LobsError("gust needs positive peak, impulse and tick")
    ticks = max(1, math.ceil(impulse / (peak * tick_dt) - 1e-9))
    return ticks, impulse / (ticks * tick_dt)


def inject_gust(sim, actor: str, direction, peak: float, impulse: float) -> dict:
    """Push ``actor`` along ``direction`` with a hard budget; returns what was delivered."""
    d = np.asarray(direction, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        raise LobsError("gust direction must be non-zero")
    ticks, mag = gust_plan(peak, impulse, sim.tick_dt)
    sim.apply_force(actor, d / n * mag, ticks=ticks)
    return {"ticks": ticks, "force": mag, "impulse": mag * ticks * sim.tick_dt, "seconds": ticks * sim.tick_dt}
