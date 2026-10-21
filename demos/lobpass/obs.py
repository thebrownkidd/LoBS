"""Observation layout (A9): 109 float32, fixed scaling, clipped to +-5, court frame."""

from __future__ import annotations

import hashlib

import numpy as np

from . import constants as C

# dunker machine state index -> one-hot slot (wait, run_c, run_t, jump/air, land/blocked)
_STATE_SLOT = (0, 1, 2, 3, 3, 4, 4)
_DEF_FIELDS = (
    "present", "x", "y", "vx", "vy", "jz", "vz", "H", "R_s", "L",
    "role_static", "role_jumper", "role_onball", "role_chaser", "role_cutoff", "role_rim",
)  # fmt: skip


def _names() -> list[str]:
    n = ["t", "n_def", "hold_x", "hold_y", "hold_z", "passer_H"]
    n += [
        "dk_H", "dk_L", "dk_R_s", "dk_x", "dk_y", "dk_jz", "dk_vx", "dk_vy", "dk_vz",
        "dk_st_wait", "dk_st_run_c", "dk_st_run_t", "dk_st_air", "dk_st_down",
        "has_corner", "corner_x", "corner_y", "takeoff_x", "takeoff_y", "vmax", "v_gather", "d_takeoff", "blocked",
    ]  # fmt: skip
    for k in range(C.N_DEF_SLOTS):
        n += [f"def{k}_{f}" for f in _DEF_FIELDS]
    return n


NAMES = _names()
assert len(NAMES) == C.OBS_DIM
OBS_VERSION = hashlib.sha1(",".join(NAMES).encode()).hexdigest()[:8]


def build(frame, slots, scn, dunker_state: int, hold_xyz, t: float) -> np.ndarray:
    """Observation from a ``StateFrame`` (positions: x, y, and jz for players), the scenario's fixed facts and the
    dunker machine state."""
    pos, vel = frame.body_pos, frame.body_linvel
    out = np.zeros(C.OBS_DIM, dtype=np.float64)
    out[0], out[1] = t / 3.0, scn.n_def / 5.0
    out[2], out[3], out[4], out[5] = (
        hold_xyz[0] / 10,
        hold_xyz[1] / 10,
        hold_xyz[2] / 3,
        scn.passer_H / 2.2,
    )
    dp, dv = pos[slots["dunker"]], vel[slots["dunker"]]
    T = np.array(scn.takeoff)
    out[6:9] = scn.dunker_H / 2.2, scn.dunker_L, scn.dunker_R_s / 3
    out[9:15] = dp[0] / 10, dp[1] / 10, dp[2], dv[0] / 8, dv[1] / 8, dv[2] / 5
    out[15 + _STATE_SLOT[dunker_state]] = 1.0
    out[20] = 1.0 if scn.corner is not None else 0.0
    cx, cy = scn.corner if scn.corner is not None else (0.0, 0.0)
    out[21:25] = cx / 10, cy / 10, T[0] / 10, T[1] / 10
    out[25], out[26] = scn.vmax / 8, scn.v_gather / 8
    out[27] = float(np.linalg.norm(dp[:2] - T)) / 10
    out[28] = 1.0 if dunker_state == 6 else 0.0
    live = sorted(
        scn.defenders, key=lambda d: float(np.linalg.norm(pos[slots[f"d{d.slot}"]][:2] - T))
    )
    for k, d in enumerate(live):
        p, v = pos[slots[f"d{d.slot}"]], vel[slots[f"d{d.slot}"]]
        o = 29 + k * C.DEF_OBS
        out[o : o + 10] = (
            1.0,
            p[0] / 10,
            p[1] / 10,
            v[0] / 8,
            v[1] / 8,
            p[2],
            v[2] / 5,
            d.H / 2.2,
            d.R_s / 3,
            d.L,
        )
        if d.role >= 1:
            out[o + 9 + d.role] = 1.0
    return np.clip(out, -5.0, 5.0).astype(np.float32)
