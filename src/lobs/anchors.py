"""Anchor rules: pure-kinematics offsets that move an anchor relative to its parent."""

from __future__ import annotations

import numpy as np

_AX = {"x": 0, "y": 1, "z": 2}


def rule_offset(rule, t: float) -> np.ndarray:
    """Offset (parent-local frame) added to the anchor's local position at time t."""
    kind = getattr(rule, "type", "rigid")
    if kind != "path":
        return np.zeros(3)
    if rule.kind == "line":
        return np.asarray(rule.velocity, dtype=float) * t
    if rule.kind == "circle":
        # orbit around the authored position, in the plane normal to `axis`;
        # at t=0 the offset points along the first in-plane axis (z -> x, x -> y, y -> z)
        i = _AX[rule.axis]
        a, b = (i + 1) % 3, (i + 2) % 3
        ph = 2.0 * np.pi * t / rule.period
        off = np.zeros(3)
        off[a], off[b] = rule.radius * np.cos(ph), rule.radius * np.sin(ph)
        return off
    if rule.kind == "sine":
        off = np.zeros(3)
        off[_AX[rule.axis]] = rule.amplitude * np.sin(2.0 * np.pi * rule.frequency * t)
        return off
    if rule.kind == "keyframes":
        kf = rule.keyframes
        if not kf:
            return np.zeros(3)
        ts = np.array([k[0] for k in kf])
        ps = np.array([k[1] for k in kf], dtype=float)
        if rule.loop and ts[-1] > ts[0]:
            t = ts[0] + (t - ts[0]) % (ts[-1] - ts[0])
        return np.array([np.interp(t, ts, ps[:, c]) for c in range(3)]) - ps[0]
    raise ValueError(f"unknown path kind {rule.kind}")
