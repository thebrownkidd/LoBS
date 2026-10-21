"""Dunker path planner (A6): pick a takeoff point and at most one corner that is short and clear of defenders.

Deterministic and vectorized: every (takeoff, corner) candidate is scored at once; ties go to the first candidate
in enumeration order (corner-less candidates first, takeoffs ordered by radius then angle)."""

from __future__ import annotations

import math

import numpy as np

from . import constants as C
from . import court


def _seg_dist(p, a, b):
    """Distance from points p (..., 2) to segments a-b (broadcast)."""
    ab = b - a
    t = np.clip(
        np.einsum("...k,...k->...", p - a, ab)
        / np.maximum(np.einsum("...k,...k->...", ab, ab), 1e-12),
        0.0,
        1.0,
    )
    return np.linalg.norm(p - (a + t[..., None] * ab), axis=-1)


def takeoff_candidates() -> np.ndarray:
    pts = []
    for r in C.PLAN_TAKEOFF_R:
        for deg in C.PLAN_ANGLES_DEG:
            pts.append(court.polar(r, math.radians(deg)))
    return np.array(pts)


def clearance(path_pts: list[np.ndarray], defenders: np.ndarray) -> np.ndarray:
    """Min distance from each defender start (n,2) to the polyline through ``path_pts``."""
    d = np.full(len(defenders), np.inf)
    for a, b in zip(path_pts[:-1], path_pts[1:], strict=True):
        d = np.minimum(d, _seg_dist(defenders, np.asarray(a), np.asarray(b)))
    return d


def plan(start, defenders, vmax: float):
    """Best (takeoff (2,), corner (2,) or None, cost). ``defenders``: (n,2) start positions (n may be 0)."""
    start = np.asarray(start, float)
    defenders = np.asarray(defenders, float).reshape(-1, 2)
    ts = takeoff_candidates()  # (nT, 2)
    lo = np.minimum(start, ts.min(axis=0)) - C.PLAN_BOX_GROW
    hi = np.maximum(start, ts.max(axis=0)) + C.PLAN_BOX_GROW
    gx = np.arange(math.floor(lo[0] / C.PLAN_GRID) * C.PLAN_GRID, hi[0] + 1e-9, C.PLAN_GRID)
    gy = np.arange(math.floor(lo[1] / C.PLAN_GRID) * C.PLAN_GRID, hi[1] + 1e-9, C.PLAN_GRID)
    corners = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
    ok = np.array([court.in_bounds(x, y, C.BOUNDS_MARGIN) for x, y in corners])
    ok &= np.linalg.norm(corners, axis=1) >= C.CORNER_MIN_RIM_DIST
    ok &= np.linalg.norm(corners - start, axis=1) >= 1.0
    corners = corners[ok]
    t_ok = np.array([court.in_bounds(x, y, C.BOUNDS_MARGIN) for x, y in ts])
    nT, nC, nD = len(ts), len(corners), len(defenders)
    straight_len = np.linalg.norm(ts - start, axis=1)  # (nT,)

    def pen(c):
        return C.PLAN_W_CLEAR * np.sum(np.maximum(0.0, C.PLAN_CLEARANCE - c) ** 2, axis=-1)

    # --- corner-less candidates (index 0..nT-1)
    if nD:
        c0 = np.stack(
            [_seg_dist(defenders[None, :, :], start[None, None, :], ts[:, None, :])], axis=0
        )[0]  # (nT, nD)
    else:
        c0 = np.zeros((nT, 0))
    cost0 = straight_len / vmax + pen(c0)
    cost0 = np.where(t_ok, cost0, np.inf)

    # --- one-corner candidates (nT x nC)
    S = start[None, None, :]
    T = ts[:, None, :]
    Cc = corners[None, :, :]
    l1 = np.linalg.norm(Cc - S, axis=-1)
    l2 = np.linalg.norm(T - Cc, axis=-1)
    length = l1 + l2
    v1 = (Cc - S) / np.maximum(l1[..., None], 1e-9)
    v2 = (T - Cc) / np.maximum(l2[..., None], 1e-9)
    turn = np.degrees(np.arccos(np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)))
    valid = (
        (turn <= C.PLAN_MAX_TURN_DEG)
        & (length <= C.PLAN_MAX_LEN_FACTOR * straight_len[:, None])
        & (l2 >= 1.0)
    )
    valid &= t_ok[:, None]
    if nD:
        D = defenders[None, None, :, :]
        d1 = _seg_dist(D, S[:, :, None, :], Cc[:, :, None, :])
        d2 = _seg_dist(D, Cc[:, :, None, :], T[:, :, None, :])
        clr = np.minimum(d1, d2)  # (nT, nC, nD)
    else:
        clr = np.zeros((nT, nC, 0))
    cost1 = length / vmax + pen(clr) + C.PLAN_W_TURN * turn / 90.0
    cost1 = np.where(valid, cost1, np.inf)

    flat = np.concatenate([cost0, cost1.reshape(-1)])
    k = int(np.argmin(flat))
    if not np.isfinite(flat[k]):
        raise ValueError("no feasible dunker path")
    if k < nT:
        return ts[k].copy(), None, float(flat[k])
    k -= nT
    return ts[k // nC].copy(), corners[k % nC].copy(), float(flat[nT + k])
