"""Scenario dataclass and per-stage seeded samplers (A5). Every sampler takes only a seed, so a scenario is a pure
function of (stage, seed)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from . import constants as C
from . import court, planner
from .bodies import apex_time, jump_cmd_vz, min_dunk_lift, reach

ROLE_OFF, ROLE_STATIC, ROLE_JUMPER, ROLE_ONBALL, ROLE_CHASER, ROLE_CUTOFF, ROLE_RIM = range(7)
ROLE_NAMES = ("off", "static", "jumper", "onball", "chaser", "cutoff", "rim")


@dataclass
class DefenderSpec:
    slot: int
    role: int
    start: tuple
    H: float
    L: float
    speed: float
    react: float

    @property
    def R_s(self) -> float:
        return reach(self.H)


@dataclass
class Scenario:
    stage: str
    seed: int
    passer_xy: tuple
    passer_H: float
    dunker_start: tuple
    dunker_H: float
    dunker_L: float
    vmax: float
    v_corner: float
    v_gather: float
    start_delay: float
    takeoff: tuple
    corner: tuple | None
    jump_vh: float
    jump_vz: float
    defenders: list[DefenderSpec] = field(default_factory=list)

    @property
    def dunker_R_s(self) -> float:
        return reach(self.dunker_H)

    @property
    def n_def(self) -> int:
        return len(self.defenders)

    def to_dict(self) -> dict:
        return asdict(self)


def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _sample_passer(rng):
    while True:
        r = _u(rng, C.PASSER_R)
        th = math.radians(rng.uniform(-C.PASSER_THETA_DEG, C.PASSER_THETA_DEG))
        p = court.polar(r, th)
        if court.in_bounds(p[0], p[1], C.BOUNDS_MARGIN):
            return p, th


def _sample_dunker_start(rng, passer_xy, passer_th):
    while True:
        r = _u(rng, C.DUNKER_START_R)
        th = math.radians(rng.uniform(-C.DUNKER_START_THETA_DEG, C.DUNKER_START_THETA_DEG))
        p = court.polar(r, th)
        if not court.in_bounds(p[0], p[1], C.BOUNDS_MARGIN):
            continue
        if abs(math.degrees(th - passer_th)) < C.DUNKER_MIN_ANGLE_DEG:
            continue
        if np.linalg.norm(p - passer_xy) < C.DUNKER_MIN_DIST_PASSER:
            continue
        return p, th


def _sample_takeoff(rng, start_th):
    r_to = _u(rng, C.TAKEOFF_R)
    ang = start_th + math.radians(
        rng.uniform(-C.TAKEOFF_ANGLE_JITTER_DEG, C.TAKEOFF_ANGLE_JITTER_DEG)
    )
    lim = math.radians(C.TAKEOFF_ANGLE_MAX_DEG)
    return court.polar(r_to, max(-lim, min(lim, ang)))


def _turn_deg(a, c, b):
    v1, v2 = c - a, b - c
    cosang = float(np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosang))))


def _sample_corner(rng, start, takeoff):
    """A corner off the straight line (or None after SAMPLE_RETRIES failed tries)."""
    line = takeoff - start
    length = float(np.linalg.norm(line))
    perp = np.array([-line[1], line[0]]) / max(length, 1e-9)
    for _ in range(C.SAMPLE_RETRIES):
        f = _u(rng, C.CORNER_FRAC)
        off = rng.choice([-1.0, 1.0]) * _u(rng, C.CORNER_OFFSET)
        c = start + f * line + off * perp
        if (
            not court.in_bounds(c[0], c[1], C.BOUNDS_MARGIN)
            or np.linalg.norm(c) < C.CORNER_MIN_RIM_DIST
        ):
            continue
        if C.CORNER_TURN_DEG[0] <= _turn_deg(start, c, takeoff) <= C.CORNER_TURN_DEG[1]:
            return c
    return None


def _jump_params(takeoff, dunker_L, v_gather):
    vz = jump_cmd_vz(dunker_L)
    t_ap = vz / C.G
    r_to = float(np.linalg.norm(takeoff))
    return min(v_gather, (r_to - C.JUMP_BACKOFF) / t_ap), vz


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


# ------------------------------------------------------------------ defenders
def _defender_start_ok(p, others, dunker_start, passer_xy):
    if not court.in_bounds(p[0], p[1], C.BOUNDS_MARGIN):
        return False
    if (
        np.linalg.norm(p - dunker_start) < C.DEF_MIN_DUNKER
        or np.linalg.norm(p - passer_xy) < C.DEF_MIN_PASSER
    ):
        return False
    return all(np.linalg.norm(p - o) >= C.DEF_MIN_SEP for o in others)


def _static_like_start(rng, kind, passer_xy, dunker_start, takeoff):
    """Placement mixture for S2/S3a (A5)."""
    if kind == "lane":
        f = rng.uniform(0.3, 0.7)
        seg = takeoff - passer_xy
        perp = _unit(np.array([-seg[1], seg[0]]))
        return passer_xy + f * seg + rng.normal(0.0, 0.5) * perp
    if kind == "rim":
        return court.polar(rng.uniform(0.8, 2.5), math.radians(rng.uniform(-70.0, 70.0)))
    if kind == "line":
        f = rng.uniform(0.3, 0.9)
        seg = takeoff - dunker_start
        perp = _unit(np.array([-seg[1], seg[0]]))
        return dunker_start + f * seg + rng.normal(0.0, 0.4) * perp
    return passer_xy + 1.1 * _unit(-passer_xy)  # onball


def _role_start(rng, role, passer_xy, dunker_start, takeoff):
    if role == ROLE_ONBALL:
        return passer_xy + 1.1 * _unit(-passer_xy)
    if role == ROLE_CHASER:
        toward = math.atan2(-dunker_start[1], -dunker_start[0]) + math.radians(
            rng.uniform(-90.0, 90.0)
        )
        return dunker_start + rng.uniform(C.DEF_MIN_DUNKER, 2.5) * np.array(
            [math.cos(toward), math.sin(toward)]
        )
    if role == ROLE_CUTOFF:
        seg = takeoff - dunker_start
        perp = _unit(np.array([-seg[1], seg[0]]))
        return (
            dunker_start
            + rng.uniform(0.4, 0.8) * seg
            + rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 4.0) * perp
        )
    return court.polar(rng.uniform(1.0, 2.0), math.radians(rng.choice([-60.0, 60.0])))  # rim


def _draw_roles(rng, n, max_rim):
    roles, onball, rim = [], 0, 0
    while len(roles) < n:
        opts = [ROLE_CHASER, ROLE_CUTOFF]
        if onball < 1:
            opts.append(ROLE_ONBALL)
        if rim < max_rim:
            opts.append(ROLE_RIM)
        role = int(rng.choice(opts))
        onball += role == ROLE_ONBALL
        rim += role == ROLE_RIM
        roles.append(role)
    return roles


def _sample_defenders(rng, stage, passer_xy, dunker_start, takeoff):
    """Returns the defender list, or None when placement failed (caller resamples the scenario)."""
    if stage in ("S2", "S3a"):
        n = int(rng.integers(1, 5))
        kinds_left, onball, roles = ["lane", "rim", "line", "onball"], 0, None
        probs_all = {"lane": 0.35, "rim": 0.25, "line": 0.25, "onball": 0.15}
    else:
        n = int(rng.integers(1, 4)) if stage == "S3b" else int(rng.integers(2, 6))
        roles = _draw_roles(rng, n, 1 if stage == "S3b" else 2)
    specs, starts = [], []
    for i in range(n):
        for _ in range(C.SAMPLE_RETRIES):
            if roles is None:
                avail = [k for k in kinds_left if k != "onball" or onball < 1]
                pr = np.array([probs_all[k] for k in avail])
                kind = avail[int(rng.choice(len(avail), p=pr / pr.sum()))]
                p = _static_like_start(rng, kind, passer_xy, dunker_start, takeoff)
                role = ROLE_STATIC if stage == "S2" else ROLE_JUMPER
            else:
                role = roles[i]
                kind = None
                p = _role_start(rng, role, passer_xy, dunker_start, takeoff)
            if _defender_start_ok(p, starts, dunker_start, passer_xy):
                break
        else:
            return None
        if kind == "onball":
            onball += 1
        if stage == "S2":
            speed, react = 0.0, 0.0
        elif stage == "S3a":
            speed, react = 0.0, _u(rng, (0.25, 0.35))
        elif stage == "S3b":
            speed, react = _u(rng, (4.5, 6.0)) * 0.8, 0.30
        else:
            speed, react = _u(rng, (4.5, 6.0)), _u(rng, (0.15, 0.25))
        specs.append(
            DefenderSpec(
                i,
                role,
                tuple(map(float, p)),
                _u(rng, C.DEF_H_RANGE),
                _u(rng, C.DEF_L_RANGE),
                speed,
                react,
            )
        )
        starts.append(p)
    return specs


# ------------------------------------------------------------------ top level
def sample_scenario(stage: str, seed: int) -> Scenario:
    if stage not in C.STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    rng = np.random.default_rng(seed)
    while True:
        passer_xy, p_th = _sample_passer(rng)
        passer_H = C.PASSER_H_S0 if stage == "S0" else _u(rng, C.PASSER_H_RANGE)
        start, s_th = _sample_dunker_start(rng, passer_xy, p_th)
        if stage == "S0":
            dH, dL = C.DUNKER_H_S0, C.DUNKER_L_S0
        else:
            dH = _u(rng, C.DUNKER_H_RANGE)
            dL = max(_u(rng, C.DUNKER_L_RANGE), min_dunk_lift(dH))
        vmax = _u(rng, C.VMAX)
        v_corner = vmax * _u(rng, C.VCORNER_FRAC)
        v_gather = _u(rng, C.VGATHER)
        delay = _u(rng, C.START_DELAY)
        takeoff = _sample_takeoff(rng, s_th)
        defenders: list[DefenderSpec] = []
        corner = None
        if stage in ("S0", "S1"):
            if rng.random() >= C.P_STRAIGHT:
                corner = _sample_corner(rng, start, takeoff)
        else:
            defenders = _sample_defenders(rng, stage, passer_xy, start, takeoff)
            if defenders is None:
                continue
            takeoff, corner, _ = planner.plan(start, [d.start for d in defenders], vmax)
        jump_vh, jump_vz = _jump_params(takeoff, dL, v_gather)
        return Scenario(
            stage=stage, seed=seed, passer_xy=tuple(map(float, passer_xy)), passer_H=passer_H,
            dunker_start=tuple(map(float, start)), dunker_H=dH, dunker_L=dL, vmax=vmax, v_corner=v_corner,
            v_gather=v_gather, start_delay=delay, takeoff=tuple(map(float, takeoff)),
            corner=None if corner is None else tuple(map(float, corner)), jump_vh=jump_vh, jump_vz=jump_vz,
            defenders=defenders,
        )  # fmt: skip


def apex(scn: Scenario) -> float:
    """Time from take-off to the dunker's apex (s)."""
    return apex_time(scn.dunker_L)
