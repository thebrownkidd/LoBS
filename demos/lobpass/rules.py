"""Event measures (A8): catch, interceptions, floor / board / rim / out, blocked, dunkable.

Every event is an SR latch gated by ``released`` (except blocked) with its time captured on the rising edge, so
the env can read them any time after the fact. Only native ops are used."""

from __future__ import annotations

from types import SimpleNamespace

import lobs
from lobs import ops, props

from . import constants as C
from .mhelp import FLAT, ball_pos, ball_vel, dimless, pos_of, released, var, vel_of

DEFS = [f"d{i}" for i in range(C.N_DEF_SLOTS)]


def _never():
    return ops.time() < -1.0


def _any(terms):
    out = terms[0]
    for t in terms[1:]:
        out = out | t
    return out


def seg_dist(b, axis3, z_lo, z_hi):
    """Distance from point b to the vertical segment over axis3 (xy used) between heights z_lo and z_hi."""
    dxy = ops.norm((b - axis3) * FLAT)
    dz = ops.max_of(0.0, z_lo - b.z, b.z - z_hi)
    return ops.norm(ops.vec(dxy, dz, 0.0))


def _event(cond, rel):
    ev = ops.latch(rel & cond, _never())
    return ev, dimless(ops.time()).sample_on(ev)


def build(w: lobs.World) -> SimpleNamespace:
    """Register the event measures on ``w``; returns the expressions other modules need (``blocked``)."""
    b, bv = ball_pos(), ball_vel()
    rel = released()
    dk_p, dk_v = pos_of("dunker"), vel_of("dunker")
    H = var("dunker", "H")

    # ---- interceptions (one latch per defender slot)
    ints = []
    for d in DEFS:
        p = pos_of(d)
        cond = (var(d, "present") > 0.5) & (
            seg_dist(b, p * FLAT, 0.0, p.z + var(d, "R_s")) < C.INT_RADIUS
        )
        ints.append(_event(cond, rel))
    int_any = _any([e for e, _ in ints])

    # ---- catch: ball inside the dunker's catch capsule, in front of him, at a catchable speed
    fwd = ops.unit(dk_v * FLAT)
    axis = dk_p * FLAT + fwd * C.CATCH_AXIS_FWD
    d_catch = seg_dist(b, axis, dk_p.z + H * C.CATCH_Z[0], dk_p.z + H * C.CATCH_Z[1])
    visible = ops.dot(fwd, ops.unit((b - dk_p) * FLAT)) >= C.CATCH_VISIBLE_DOT
    catch_now = (
        (d_catch < C.CATCH_RADIUS)
        & visible
        & (ops.norm(bv - dk_v) <= C.CATCH_MAX_REL_SPEED)
        & ~int_any
    )
    ev_catch, t_catch = _event(catch_now, rel)

    # ---- ball vs the world
    ev_floor, t_floor = _event(b.z < C.FLOOR_Z, rel)
    half = ops.vec(*(s / 2 for s in C.BOARD_SIZE))
    off = ops.clip(abs(b - ops.vec(*C.BOARD_CENTER)) - half, 0.0, None)
    ev_board, t_board = _event(ops.norm(off) < C.BALL_R, rel)
    rho = ops.norm(b * FLAT)
    ring = ops.norm(ops.vec(rho - C.RIM_RING_R, b.z - C.RIM_TUBE_Z, 0.0))
    ev_rim, t_rim = _event(ring < C.RIM_HIT_DIST, rel)
    out = (
        (b.x < C.BASELINE_X - C.OUT_MARGIN)
        | (b.x > C.HALFCOURT_X + C.OUT_MARGIN)
        | (abs(b.y) > C.SIDELINE_Y + C.OUT_MARGIN)
    )
    ev_out, t_out = _event(out, rel)

    # ---- blocked: a grounded defender is within body contact of the dunker (never gated by the throw)
    near = [
        (var(d, "present") > 0.5)
        & (pos_of(d).z < C.BLOCK_JZ)
        & (ops.norm((pos_of(d) - dk_p) * FLAT) < C.BLOCK_DIST)
        for d in DEFS
    ]
    blocked = ~ev_catch & _any(near)
    ev_blocked = ops.latch(blocked, _never())
    t_blocked = dimless(ops.time()).sample_on(ev_blocked)

    # ---- dunkable, sampled at the catch: a finishing chance exists from that body state
    air = (
        (dk_p.z >= C.DUNKABLE_AIR_JZ)
        & (b.z >= C.DUNKABLE_AIR_BALL_Z)
        & (ops.norm(dk_p * FLAT) <= C.DUNKABLE_AIR_RIM_DIST)
        & (dk_v.z >= C.DUNKABLE_AIR_VZ)
    )
    contest = [
        (var(d, "present") > 0.5)
        & (pos_of(d).z > C.BLOCK_JZ)
        & (ops.norm((pos_of(d) - dk_p) * FLAT) < C.DUNKABLE_AIR_DEF_DIST)
        & (pos_of(d).z + var(d, "R_s") >= b.z - 0.1)
        for d in DEFS
    ]
    air = air & ~_any(contest)
    crowd = [
        (var(d, "present") > 0.5)
        & (ops.norm((pos_of(d) - dk_p) * FLAT) < C.DUNKABLE_GROUND_DEF_DIST)
        for d in DEFS
    ]
    ground = (
        props.behavior_in_state("dunker", "run_t")
        & (ops.norm((var("dunker", "takeoff") - dk_p) * FLAT) <= C.DUNKABLE_GROUND_T_DIST)
        & ~_any(crowd)
    )
    airborne = dk_p.z >= C.DUNKABLE_AIR_JZ
    dunkable_now = (airborne & air) | (~airborne & ground)
    dunkable = ops.where(dunkable_now, 1.0, 0.0).sample_on(ev_catch)

    # ---- closest approach to the catch capsule after release (reward shaping on misses)
    dmin = ops.where(rel, d_catch, 100.0).running_min()

    for name, m in {
        "ev_catch": ev_catch, "t_catch": t_catch, "ev_floor": ev_floor, "t_floor": t_floor,
        "ev_board": ev_board, "t_board": t_board, "ev_rim": ev_rim, "t_rim": t_rim,
        "ev_out": ev_out, "t_out": t_out, "ev_blocked": ev_blocked, "t_blocked": t_blocked,
        "dunkable": dunkable, "catch_dmin": dmin, "catch_d": d_catch,
    }.items():  # fmt: skip
        w.measure(name, m)
    for i, (e, t) in enumerate(ints):
        w.measure(f"ev_int{i}", e)
        w.measure(f"t_int{i}", t)
    return SimpleNamespace(blocked=blocked)
