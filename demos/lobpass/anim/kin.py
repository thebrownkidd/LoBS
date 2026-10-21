"""Base kinematics of every character on the 240 Hz bake timeline: root path, facing, air/crouch/lean signals and
the planted-foot gait. Arms, ball and camera are built on top of this (track.py)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import constants as C
from . import gait

FPS = 240  # internal bake rate; video frames are sampled from it
SIM_HZ = 100
G = C.G


def tick_to_frame(tick: float) -> int:
    return int(round(tick * FPS / SIM_HZ))


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average along axis 0 with edge padding."""
    win = max(1, int(win) | 1)
    pad = win // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x, np.repeat(x[-1:], pad, 0)], 0)
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), xp], 0), 0)
    return (c[win:] - c[:-win]) / win


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def spring_angle(target: np.ndarray, omega: float = 12.0, max_rate: float = 10.0) -> np.ndarray:
    """Critically damped follower of an angle series (unwrapped), rate-limited."""
    tgt = np.unwrap(target)
    dt = 1.0 / FPS
    x, v = float(tgt[0]), 0.0
    out = np.empty_like(tgt)
    for k, t in enumerate(tgt):
        a = omega * omega * (t - x) - 2.0 * omega * v
        v = float(np.clip(v + a * dt, -max_rate, max_rate))
        x += v * dt
        out[k] = x
    return out


def ffill_angle(a: np.ndarray, valid: np.ndarray, default: float) -> np.ndarray:
    out = a.copy()
    last = default
    for k in range(len(out)):
        if valid[k]:
            last = out[k]
        out[k] = last
    if valid.any():  # leading invalid frames take the first valid heading
        first = np.argmax(valid)
        out[:first] = a[first]
    return out


@dataclass
class Sim:
    """Episode resampled to FPS: positions (K,3) for 'ball' and each player, velocities, plus the metadata."""

    K: int
    pos: dict
    vel: dict
    meta: dict
    events: list
    dunker_state: np.ndarray = None
    def_state: np.ndarray = None

    def event(self, kind, actor=None):
        for e in self.events:
            if e["type"] == kind and (actor is None or e["actor"] == actor):
                return e
        return None


def load_sim(arrays: dict, meta: dict) -> Sim:
    ticks = arrays["tick"].astype(float)
    n = len(ticks)
    t_old = ticks / SIM_HZ
    K = int(round((n - 1) * FPS / SIM_HZ)) + 1
    t_new = np.arange(K) / FPS
    names = ["ball"] + list(meta["players"])
    pos, vel = {}, {}
    for j, name in enumerate(names):
        pos[name] = np.stack([np.interp(t_new, t_old, arrays["pos"][:, j, d]) for d in range(3)], 1)
        vel[name] = np.stack([np.interp(t_new, t_old, arrays["vel"][:, j, d]) for d in range(3)], 1)
    idx = np.minimum(np.round(t_new * SIM_HZ).astype(int), n - 1)
    return Sim(
        K, pos, vel, meta, meta["events"], arrays["dunker_state"][idx], arrays["def_state"][idx]
    )


@dataclass
class Kin:
    id: str
    team: str
    role: str
    H: float
    L: float
    root_xy: np.ndarray
    jz: np.ndarray
    vel_xy: np.ndarray
    speed: np.ndarray
    yaw: np.ndarray
    air: np.ndarray
    shuffle: np.ndarray
    feet: dict
    crouch: np.ndarray  # pelvis drop in metres (>=0)
    pelvis_dz: np.ndarray  # gait bob in metres
    lean: np.ndarray
    roll: np.ndarray
    counter: np.ndarray
    takeoffs: list = field(default_factory=list)
    lands: list = field(default_factory=list)


def _segments(mask: np.ndarray):
    """(start, end) frame index pairs of the contiguous True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0], strict=True))


def build_kin(sim: Sim, actor: dict, facing_desired: np.ndarray) -> Kin:
    """Kinematics of one actor. ``facing_desired`` (K,) is the heading the character wants (before smoothing)."""
    name, H = actor["id"], actor["H"]
    root = sim.pos[name][:, :2].copy()
    jz = np.maximum(sim.pos[name][:, 2], 0.0)
    vel = smooth(np.gradient(smooth(root, 9), axis=0) * FPS, 25)
    speed = np.linalg.norm(vel, axis=1)
    yaw = spring_angle(facing_desired)
    air = jz > 0.03
    head = np.stack([np.cos(yaw), np.sin(yaw)], 1)
    vhat = vel / np.maximum(speed[:, None], 1e-6)
    ang = np.arccos(np.clip(np.sum(head * vhat, 1), -1, 1))
    shuffle = (speed > 0.5) & (np.degrees(ang) > 60.0) & (actor["team"] == "defense")
    side = np.where(actor["team"] == "defense", np.where(speed < 2.5, 0.15, 0.11), gait.SIDE)
    feet = gait.plan_feet(root, vel, yaw, air, H, FPS, shuffle, side)
    # --- pelvis signals
    phase = feet["phase"]
    amp = 0.025 * H * np.minimum(speed / 5.0, 1.0)
    bob = -amp * np.cos(4 * math.pi * phase)
    acc = np.gradient(smooth(vel, 25), axis=0) * FPS
    a_fwd = np.sum(acc * head, 1)
    lean = np.clip(
        0.6 * np.arctan(a_fwd / G) + np.radians(12.0) * np.minimum(speed / 6.0, 1.0),
        np.radians(-5.0),
        np.radians(22.0),
    )
    heading = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    heading = smooth(heading, 25)
    omega = np.gradient(heading) * FPS
    roll = np.clip(np.arctan(speed * omega / G), -math.radians(25), math.radians(25)) * (
        speed > 1.0
    )
    counter = np.radians(8.0) * np.minimum(speed / 6.0, 1.0) * np.sin(2 * math.pi * phase)
    # --- jumps: crouch before take-off, absorb on landing
    crouch = np.zeros(sim.K)
    takeoffs, lands = [], []
    for s, e in _segments(air):
        takeoffs.append(int(s))
        lands.append(int(e))
        pre = int(0.15 * FPS)
        for k in range(max(0, s - pre), s):
            crouch[k] = max(crouch[k], 0.08 * H * smooth_step((k - (s - pre)) / pre))
        for k in range(e, min(sim.K, e + int(0.37 * FPS))):
            t = (k - e) / FPS
            c = (
                0.10
                * H
                * (smooth_step(t / 0.06) if t < 0.06 else 1.0 - smooth_step((t - 0.06) / 0.31))
            )
            crouch[k] = max(crouch[k], c)
    return Kin(name, actor["team"], actor["role"], H, actor["L"], root, jz, vel, speed, yaw, air, shuffle, feet,
               crouch, bob, lean, roll, counter, takeoffs, lands)  # fmt: skip


def smooth_step(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)
