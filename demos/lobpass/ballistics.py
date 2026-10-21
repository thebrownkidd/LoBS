"""Ball flight with quadratic drag, matching the LoBS tick: drag is a zero-order hold over the tick's substeps and
MuJoCo integrates semi-implicitly (velocity first, then position) at dt = C.DT.

Also the drag-free action inverse the env uses, and a drag-aware shooting solver for the oracle."""

from __future__ import annotations

import math

import numpy as np

from . import constants as C

_GRAV = np.array([0.0, 0.0, -C.G])


def tick(p: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One LoBS tick of free flight (drag held over the substeps)."""
    a = _GRAV - (C.DRAG_C / C.BALL_MASS) * float(np.linalg.norm(v)) * v
    p, v = p.copy(), v.copy()
    for _ in range(C.SUBSTEPS):
        v = v + a * C.DT
        p = p + v * C.DT
    return p, v


def simulate(p0, v0, ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """Positions and velocities after 0..ticks ticks: arrays (ticks+1, 3)."""
    p, v = np.asarray(p0, float), np.asarray(v0, float)
    ps, vs = [p], [v]
    for _ in range(ticks):
        p, v = tick(p, v)
        ps.append(p)
        vs.append(v)
    return np.array(ps), np.array(vs)


def drag_free_v0(z0: float, psi: float, h_apex: float, range_xy: float) -> tuple[np.ndarray, float]:
    """Launch velocity for apex height h_apex above the floor, landing range_xy away at rim height (A9 inverse).
    Returns (v0, flight_time)."""
    vz = math.sqrt(2.0 * C.G * max(h_apex - z0, 1e-6))
    t_f = vz / C.G + math.sqrt(2.0 * max(h_apex - C.RIM_TOP_Z, 1e-6) / C.G)
    vh = range_xy / t_f
    return np.array([vh * math.cos(psi), vh * math.sin(psi), vz]), t_f


def clamp_speed(v0: np.ndarray) -> tuple[np.ndarray, bool]:
    """Scale the horizontal part down so |v0| <= MAX_THROW_SPEED (A3). Returns (v0, clamped)."""
    s = float(np.linalg.norm(v0))
    if s <= C.MAX_THROW_SPEED:
        return v0, False
    vz = v0[2]
    room = C.MAX_THROW_SPEED**2 - vz * vz
    if room <= 0.0:
        return v0 * (C.MAX_THROW_SPEED / s), True
    vh = float(np.linalg.norm(v0[:2]))
    out = v0.copy()
    out[:2] *= math.sqrt(room) / max(vh, 1e-9)
    return out, True


def solve_v0(p0, target, ticks: int, iters: int = 4):
    """Launch velocity reaching ``target`` after exactly ``ticks`` ticks, with drag (Newton, finite-difference
    Jacobian). Returns (v0, miss_distance)."""
    p0, target = np.asarray(p0, float), np.asarray(target, float)
    t = ticks * C.TICK
    v = (target - p0) / t + 0.5 * C.G * t * np.array([0.0, 0.0, 1.0])
    for _ in range(iters):
        end = simulate(p0, v, ticks)[0][-1]
        err = target - end
        if np.linalg.norm(err) < 1e-4:
            break
        jac = np.empty((3, 3))
        for k in range(3):
            dv = np.zeros(3)
            dv[k] = 1e-3
            jac[:, k] = (simulate(p0, v + dv, ticks)[0][-1] - end) / 1e-3
        v = v + np.linalg.solve(jac, err)
    miss = float(np.linalg.norm(target - simulate(p0, v, ticks)[0][-1]))
    return v, miss
