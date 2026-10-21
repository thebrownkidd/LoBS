"""Behavior state machines (A7). Native ops only: no script behaviors, no contacts, no path anchors.

State order matters: the index of a state in STATES is what ``props.behavior_state`` reports."""

from __future__ import annotations

import lobs
from lobs import act, ops

from . import constants as C
from .mhelp import FLAT, always, body_path, dimless, newton, pos_of, var, vel_of

DUNKER_STATES = ("wait", "run_c", "run_t", "jump", "air", "land", "blocked")
DEFENDER_STATES = ("dispatch", "off", "static")  # L7 appends the rest


def move_force(player: str, desired_v):
    """Acceleration-limited velocity tracking in the floor plane: F = m clip_norm(gain (v_des - v), a_max)."""
    v = vel_of(player) * FLAT
    return newton(ops.clip_norm((desired_v - v) * C.MOVE_GAIN, C.MOVE_MAX_ACCEL) * C.PLAYER_MASS)


def arrive_speed(dist, speed):
    """Desired speed when arriving: min(speed, 0.8 sqrt(2 a_max max(dist - 0.05, 0)))."""
    return ops.min_of(speed, (ops.max_of(dist - 0.05, 0.0) * (2.0 * C.MOVE_MAX_ACCEL)) ** 0.5 * 0.8)


def arrive_force(player: str, target, speed):
    d = (target - pos_of(player)) * FLAT
    return move_force(player, ops.unit(d) * arrive_speed(ops.norm(d), speed))


def hold_action(player: str):
    return act.steer(
        mode="hold", on=body_path(player), max_accel=C.MOVE_MAX_ACCEL, gain=C.MOVE_GAIN
    )


def build_dunker(w: lobs.World, blocked=None) -> None:
    me = "dunker"
    body = body_path(me)
    p, v = pos_of(me), vel_of(me)
    d_c = (var(me, "corner") - p) * FLAT
    d_t = (var(me, "takeoff") - p) * FLAT
    dist_c, dist_t = ops.norm(d_c), ops.norm(d_t)
    vmax = var(me, "vmax")
    s_c = ops.min_of(
        vmax, var(me, "v_corner") + ops.max_of(dist_c - C.CORNER_RADIUS, 0.0) * C.APPROACH_GAIN
    )
    s_t = ops.min_of(
        vmax, var(me, "v_gather") + ops.max_of(dist_t - C.APPROACH_SLOW, 0.0) * C.APPROACH_GAIN
    )
    # jump impulse (A4): horizontal velocity toward the rim, vertical take-off speed, one tick
    toward_rim = ops.unit(p * FLAT * -1.0)
    vh = toward_rim * var(me, "jump_vh")
    jump_f = newton(
        (vh - v * FLAT) * (C.PLAYER_MASS / C.TICK)
        + ops.vec(0, 0, (var(me, "jump_vz") / C.TICK + C.G) * C.PLAYER_MASS)
    )

    sm = lobs.StateMachine("dunker")
    sm.var("corner", (0.0, 0.0, 0.0)).var("takeoff", (1.3, 0.0, 0.0))
    for name, val in dict(has_corner=0.0, vmax=6.0, v_corner=4.0, v_gather=3.5, start_delay=1e6, jump_vz=3.9,
                          jump_vh=1.0, H=C.DUNKER_H_S0, R_s=C.reach(C.DUNKER_H_S0), L=C.DUNKER_L_S0, present=1.0).items():  # fmt: skip
        sm.var(name, val)
    sm.state("wait", do=[hold_action(me)])
    sm.state("run_c", do=[act.force(move_force(me, ops.unit(d_c) * s_c), on=body)])
    sm.state("run_t", do=[act.force(move_force(me, ops.unit(d_t) * s_t), on=body)])
    sm.state("jump", on_enter=[act.force(jump_f, on=body)])
    sm.state("air")
    sm.state("land", do=[hold_action(me)])
    sm.state("blocked", do=[hold_action(me)])
    go = dimless(ops.time()) >= var(me, "start_delay")
    has_c = var(me, "has_corner") > 0.5
    sm.transition("wait", "run_c", when=go & has_c)
    sm.transition("wait", "run_t", when=go & ~has_c)
    sm.transition("run_c", "run_t", when=dist_c < C.CORNER_RADIUS)
    sm.transition("run_t", "jump", when=dist_t < C.TAKEOFF_RADIUS, min_time=0.05)
    sm.transition("jump", "air", when=always(), min_time=C.TICK)
    sm.transition("air", "land", when=p.z < 0.003, min_time=0.1)
    if blocked is not None:
        for s in ("wait", "run_c", "run_t"):
            sm.transition(s, "blocked", when=blocked, priority=10)
    w.add_behavior("dunker", on="dunker", spec=sm, period=C.DUNKER_PERIOD)


def build_defender(w: lobs.World, i: int) -> None:
    me = f"d{i}"
    body = body_path(me)
    home = var(me, "home")
    role = var(me, "role")
    sm = lobs.StateMachine(me)
    sm.var("home", (0.0, 0.0, 0.0))
    for name, val in dict(present=0.0, role=0.0, speed=0.0, R_s=C.reach(2.0), L=0.7, vz0=3.7, react=0.3, jump_ready=1.0).items():  # fmt: skip
        sm.var(name, val)
    sm.state("dispatch")
    sm.state("off")
    sm.state("static", do=[act.force(arrive_force(me, home, 2.0), on=body)])
    sm.transition("dispatch", "off", when=role.eq(0.0))
    sm.transition("dispatch", "static", when=role.eq(1.0))
    w.add_behavior(me, on=me, spec=sm, period=C.DEF_PERIOD, latency=C.DEF_LATENCY)
