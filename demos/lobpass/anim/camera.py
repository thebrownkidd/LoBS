"""Camera director (B6) and slow-motion time remap, baked per internal frame.

Shots: broadcast (sideline, spring-follow), release (over the passer's shoulder), baseline-low slow motion for the
decisive moment. Hard cuts only, never inside the protected windows around release and catch."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import kin
from .ik import smoothstep as ik_smooth
from .kin import FPS

BROADCAST = np.array([5.0, -12.6, 4.4])
FOV = {"broadcast": 34.0, "release": 42.0, "baseline": 46.0}
PROTECT = 0.25  # s, no cut within this window of release or catch
RAMP = 0.15  # s, speed ramps
SLOW = 0.25  # slow-motion factor


@dataclass
class Direction:
    pos: np.ndarray  # (K,3) per internal frame
    target: np.ndarray
    fov: np.ndarray
    shots: list  # (f0, f1, name)
    cuts: list  # frame indices where the shot changes
    slow: tuple | None  # (sim_t0, sim_t1) or None
    protected: list = field(default_factory=list)  # (f0, f1) windows where no cut is allowed


def _spring_follow(pts: np.ndarray, omega: float = 2.5) -> np.ndarray:
    out = np.empty_like(pts)
    x, v = pts[0].copy(), np.zeros(3)
    dt = 1.0 / FPS
    for k, t in enumerate(pts):
        a = omega * omega * (t - x) - 2.0 * omega * v
        v = v + a * dt
        x = x + v * dt
        out[k] = x
    return out


def direct(baked) -> Direction:
    sim, bp = baked.sim, baked.plan
    K = sim.K
    ball = bp.pos
    rim = np.array([0.0, 0.0, 3.05])
    dunker = sim.pos["dunker"] + np.array([0.0, 0.0, 1.0])
    mid = (ball + dunker + rim) / 3.0
    follow = _spring_follow(mid)
    # --- shot list (sim-time -> frames)
    sec = lambda s: int(round(s * FPS))  # noqa: E731
    shots: list[tuple[int, int, str]] = []
    protected: list[tuple[int, int]] = []
    slow = None
    f_rel = bp.f_release
    f_event = bp.f_slam or bp.f_catch or bp.f_intercept
    if f_event is None and f_rel is not None:  # a miss: the first rim / board / floor touch
        for e in sim.events:
            if e["type"] in ("rim_touch", "board_touch", "floor_touch"):
                f_event = kin.tick_to_frame(e["tick"])
                break
    rel_end = None if f_rel is None else min(f_rel + sec(0.3), K)
    base0 = base1 = None
    if f_event is not None and f_rel is not None:
        f_ev0 = bp.f_catch if bp.f_catch is not None else f_event
        base0 = f_ev0 - sec(0.35)
        base1 = min((bp.f_slam or f_event) + sec(0.4), K)
        if base1 - base0 >= sec(0.5):
            slow = (
                base0 / FPS,
                base1 / FPS,
            )  # slow motion for the decisive moment, with or without the cut
        if base0 - rel_end < sec(0.25) or base1 - base0 < sec(0.5):
            base0 = base1 = (
                None  # too short for a separate baseline shot: stay on the broadcast camera
            )
    if f_rel is not None:
        shots.append((max(f_rel - sec(0.4), 0), rel_end, "release"))
        protected.append((f_rel - sec(PROTECT), f_rel + sec(PROTECT)))
        if bp.f_catch is not None:
            protected.append((bp.f_catch - sec(PROTECT), bp.f_catch + sec(PROTECT)))
    if base0 is not None:
        shots.append((base0, base1, "baseline"))
    shots.sort()
    # broadcast fills the gaps
    cur, full = 0, []
    for a, b, name in shots:
        if a > cur:
            full.append((cur, a, "broadcast"))
        full.append((a, b, name))
        cur = b
    if cur < K:
        full.append((cur, K, "broadcast"))
    pos, tgt, fov = np.zeros((K, 3)), np.zeros((K, 3)), np.zeros(K)
    p_end = sim.pos["passer"]
    for a, b, name in full:
        for f in range(a, b):
            if name == "broadcast":
                pos[f], tgt[f] = BROADCAST, follow[f]
            elif name == "release":
                d = bp.pass_dir if bp.pass_dir is not None else 0.0
                back = np.array([-math.cos(d), -math.sin(d), 0.0])
                side = np.array([-math.sin(d), math.cos(d), 0.0])
                pos[f] = p_end[f] + back * 4.4 + side * 1.3 + np.array([0, 0, 3.4])
                ahead = (
                    p_end[f]
                    + np.array([math.cos(d), math.sin(d), 0.0]) * 3.0
                    + np.array([0, 0, 2.6])
                )
                w_lead = ik_smooth(
                    (f - (bp.f_release - sec(0.15))) / sec(0.3)
                )  # ease into following the ball
                lead = (1 - w_lead) * ball[f] + w_lead * ball[min(f + sec(0.25), K - 1)]
                tgt[f] = 0.55 * lead + 0.45 * ahead
            else:  # baseline low
                ref = sim.pos["dunker"][bp.f_catch or f]
                sgn = 1.0 if ref[1] >= 0 else -1.0
                pos[f] = np.array([-4.8, sgn * 4.6, 1.5])
                tgt[f] = (
                    (ball[f] + np.array([0, 0, 0.2]) + dunker[f]) / 2
                    if bp.f_slam is None
                    else (ball[f] * 0.6 + rim * 0.4)
                )
            fov[f] = FOV[name]
    for a, b, _ in full:
        limit_pan(pos, tgt, a, b)
    cuts = [a for a, _, _ in full[1:]]
    return Direction(pos, tgt, fov, full, cuts, slow, protected)


def limit_pan(pos: np.ndarray, tgt: np.ndarray, a: int, b: int, max_deg_s: float = 55.0) -> None:
    """Slew-limit the view direction inside one shot (frames a..b) so the camera never pans faster than
    ``max_deg_s`` (in sim time); the look-at distance is kept."""
    step = math.radians(max_deg_s) / FPS
    prev = None
    for f in range(a, b):
        d = tgt[f] - pos[f]
        dist = float(np.linalg.norm(d))
        cur = d / max(dist, 1e-9)
        if prev is not None:
            ang = math.acos(float(np.clip(np.dot(prev, cur), -1.0, 1.0)))
            if ang > step:
                t = step / ang  # spherical interpolation from prev toward cur by the allowed angle
                s = math.sin(ang)
                cur = (math.sin((1 - t) * ang) * prev + math.sin(t * ang) * cur) / s
                cur /= np.linalg.norm(cur)
                tgt[f] = pos[f] + cur * dist
        prev = cur


def speed_curve(slow: tuple | None, K: int) -> np.ndarray:
    """Playback speed (sim seconds per video second) for each internal frame: 1x, ``SLOW`` inside the slow window,
    smoothstep ramps of RAMP seconds on either side."""
    t = np.arange(K) / FPS
    s = np.ones(K)
    if slow is None:
        return s
    t0, t1 = slow
    ramp_in = np.clip((t - (t0 - RAMP)) / RAMP, 0, 1)
    ramp_out = np.clip(((t1 + RAMP) - t) / RAMP, 0, 1)
    w = (ramp_in * ramp_in * (3 - 2 * ramp_in)) * (ramp_out * ramp_out * (3 - 2 * ramp_out))
    return 1.0 - w * (1.0 - SLOW)


def video_times(slow: tuple | None, K: int, fps_out: int = 60) -> np.ndarray:
    """Sim time (s) of each output video frame, from integrating the speed curve."""
    speed = speed_curve(slow, K)
    t_end = (K - 1) / FPS
    out, t = [], 0.0
    while t <= t_end + 1e-9:
        out.append(t)
        t += float(np.interp(t, np.arange(K) / FPS, speed)) / fps_out
    return np.array(out)
