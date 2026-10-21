"""Footstep planner (B3): phase-driven gait with planted feet fixed in world space.

A stance foot never moves: it is planted under where the root will be at mid-stance and stays at that world point
until it lifts. Swings are Bezier-like arcs from the lift point to the next plant point, which is predicted from
the root path, so the foot arrives exactly at its plant point when the phase wraps."""

from __future__ import annotations

import math

import numpy as np

from .. import constants as C  # noqa: F401  (kept for symmetry with sibling modules)

SIDE = 0.09  # lateral foot offset from the root line, x H
F_MIN = 1.6  # cycles per second while finishing a swing
V_MOVE = 0.25  # m/s below which a player stands


def duty(v: float) -> float:
    """Stance fraction of the cycle: 0.60 below 2 m/s, 0.38 above 4.5 m/s, linear between."""
    return float(np.interp(v, [2.0, 4.5], [0.60, 0.38]))


def step_len(v: float, H: float, shuffle: bool) -> float:
    if shuffle:
        return 0.40 * H
    return float(np.clip((H / 1.9) * (0.45 + 0.24 * v), 0.35 * H, 1.15 * H))


def _perp(d):
    return np.array([-d[1], d[0]])


def plan_feet(root_xy, vel_xy, yaw, air, H: float, fps: float, shuffle=None, side_h=None):
    """Foot trajectories for one player.

    root_xy (T,2), vel_xy (T,2), yaw (T,), air (T,) bool, shuffle (T,) bool or None.
    Returns dict: foot (T,2,3) world positions of the ankle-level foot point (z = height above the floor of the
    sole), stance (T,2) bool (planted on the ground), swing_u (T,2) in [0,1] or -1, foot_yaw (T,2)."""
    T = len(root_xy)
    dt = 1.0 / fps
    shuffle = np.zeros(T, bool) if shuffle is None else shuffle
    side_h = np.full(T, SIDE) if side_h is None else side_h  # lateral foot offset per frame, x H
    phase_out = np.zeros(T)
    speed = np.linalg.norm(vel_xy, axis=1)
    foot = np.zeros((T, 2, 3))
    stance = np.zeros((T, 2), bool)
    swing_u = -np.ones((T, 2))
    fyaw = np.zeros((T, 2))
    P = [None, None]
    lift = [None, None]
    nxt = [None, None]
    swinging = [False, False]
    duty_i = [0.6, 0.6]
    frac0 = [0.6, 0.6]  # phase at which each foot's current swing started
    prev_frac = [0.0, 0.0]
    phase = 0.05
    grounded_prev = False

    def side_vec(k, s):
        d = (
            vel_xy[k] / max(speed[k], 1e-6)
            if speed[k] > 0.1
            else np.array([math.cos(yaw[k]), math.sin(yaw[k])])
        )
        face = np.array([math.cos(yaw[k]), math.sin(yaw[k])])
        base = _perp(face) if shuffle[k] or speed[k] <= 0.1 else _perp(d)
        return base * (side_h[k] * H) * (1.0 if s == 0 else -1.0)  # foot 0 = left

    for k in range(T):
        if air[k]:
            for i in range(2):
                if P[i] is not None:
                    foot[k, i] = P[i]
                fyaw[k, i] = yaw[k]
            grounded_prev = False
            continue
        if (
            not grounded_prev
        ):  # touch-down (or first frame): both feet plant under the root together
            for i in range(2):
                P[i] = np.array([*(root_xy[k] + side_vec(k, i)), 0.0])
                swinging[i] = False
                duty_i[i] = duty(speed[k])
                prev_frac[i] = (phase + 0.5 * i) % 1.0
            grounded_prev = True
        v = speed[k]
        sl = step_len(v, H, bool(shuffle[k]))
        far = max(float(np.linalg.norm(P[i][:2] - (root_xy[k] + side_vec(k, i)))) for i in range(2))
        moving = v > V_MOVE or far > 0.30 * H or swinging[0] or swinging[1]
        # cadence: never slower than the step length asks, and fast enough that a stance contact spans <= 0.45 H
        f = max(v / (2.0 * sl), v * duty(v) / (0.45 * H), F_MIN) if moving else 0.0
        phase = (phase + f * dt) % 1.0 if f > 0 else phase
        phase_out[k] = phase
        for i in range(2):
            frac = (phase + 0.5 * i) % 1.0
            wrapped = frac < prev_frac[i] - 1e-9
            if swinging[i] and wrapped:  # swing ends exactly on the plant point
                P[i] = nxt[i].copy()
                swinging[i] = False
                duty_i[i] = duty(v)
            behind = (
                float(np.dot(root_xy[k] - P[i][:2], vel_xy[k] / max(v, 1e-6)))
                if (not swinging[i] and v > 0.5)
                else 0.0
            )
            if (
                not swinging[i]
                and (frac >= duty_i[i] or behind > 0.22 * H)
                and f > 0
                and not wrapped
            ):
                frac0[i] = frac  # (an early toe-off keeps the hip from over-extending)
                swinging[i] = True
                lift[i] = P[i].copy()
                t_plant = (1.0 - frac) / f
                t_mid = t_plant + 0.5 * duty_i[i] / f
                idx = int(min(T - 1, k + t_mid * fps))
                nxt[i] = np.array([*(root_xy[idx] + side_vec(idx, i)), 0.0])
                k_td = int(
                    min(T - 1, k + t_plant * fps)
                )  # the plant must stay within leg reach of the root at touch-down
                off = nxt[i][:2] - root_xy[k_td]
                lim = 0.24 * H
                if np.linalg.norm(off) > lim:
                    nxt[i][:2] = root_xy[k_td] + off * (lim / np.linalg.norm(off))
            prev_frac[i] = frac
            if swinging[i]:
                u = min(max((frac - frac0[i]) / max(1.0 - frac0[i], 1e-6), 0.0), 1.0)
                e = u * u * (3 - 2 * u)
                p = lift[i] + (nxt[i] - lift[i]) * e
                apex = H * (0.04 + 0.06 * min(v / 7.0, 1.0))
                p[2] = apex * math.sin(math.pi * u)
                if shuffle[k]:  # arc around the planted foot instead of through it
                    p[:2] += _perp(np.array([math.cos(yaw[k]), math.sin(yaw[k])])) * 0.0 + np.array(
                        [math.cos(yaw[k]), math.sin(yaw[k])]
                    ) * (0.05 * H * math.sin(math.pi * u))
                foot[k, i], swing_u[k, i] = p, u
            else:
                foot[k, i] = P[i]
                stance[k, i] = True
            fyaw[k, i] = (
                yaw[k]
                if swinging[i]
                else fyaw[k - 1, i]
                if (k > 0 and stance[k - 1, i])
                else yaw[k]
            )
    return {
        "foot": foot,
        "stance": stance,
        "swing_u": swing_u,
        "foot_yaw": fyaw,
        "phase": phase_out,
    }
