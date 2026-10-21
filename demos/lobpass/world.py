"""The one LoBS world: court, hoop, ball (drag + hold), seven player rigs, passer machine.

Training and evaluation use mode="train" and must stay on the native tick core (see tests/lobpass/test_world.py).
Every player is a fixed-root actor on x/y/z slide joints; per-episode heights live in behavior variables only.
"""

from __future__ import annotations

import math

import numpy as np

import lobs
from lobs import ops, props
from lobs.schema import GeomSpec, MaterialSpec, Pose

from . import constants as C
from . import mhelp
from .bodies import jump_cmd_vz
from .mhelp import ball_pos, ball_vel, body_path, dimless, newton, pos_of, vel_of  # noqa: F401

PLAYERS = ("passer", "dunker", *[f"d{i}" for i in range(C.N_DEF_SLOTS)])
DEFENDERS = PLAYERS[2:]


def _rig(w: lobs.World, name: str, x: float, y: float):
    a = w.add_actor(name, root="fixed", pos=(0, 0, 0))
    ball_geom = lobs.Sphere(radius=C.RIG_SPHERE_R)
    kw = dict(geoms=[ball_geom], mass=C.RIG_LINK_MASS, collision_group=C.GROUP_RIG)
    base = a.add_body("base", **kw)
    sx = base.add_child("sx", joint=lobs.Slide("jx", axis=(1, 0, 0)), **kw)
    sy = sx.add_child("sy", joint=lobs.Slide("jy", axis=(0, 1, 0)), **kw)
    sy.add_child(
        "body",
        joint=lobs.Slide("jz", axis=(0, 0, 1), range=C.SLIDE_Z_RANGE),
        geoms=[
            GeomSpec(
                shape=lobs.Capsule(radius=C.PLAYER_CAPSULE_R, length=C.PLAYER_CAPSULE_LEN),
                pose=Pose(pos=(0, 0, C.PLAYER_CAPSULE_Z)),
                material=MaterialSpec(friction=C.PLAYER_FRICTION),
            )
        ],
        mass=C.PLAYER_MASS,
        collision_group=C.GROUP_PLAYER,
    )
    a.set_initial(q={"jx": x, "jy": y})
    return a


def _rim(w: lobs.World):
    seg = 2.0 * C.RIM_RING_R * math.sin(math.pi / C.RIM_SEGMENTS)
    geoms = []
    for i in range(C.RIM_SEGMENTS):
        a = 2.0 * math.pi * (i + 0.5) / C.RIM_SEGMENTS
        c, s = math.cos(a), math.sin(a)
        # capsule axis is local z; rotate it onto the ring tangent (-s, c, 0): 90 deg about (-c, -s, 0)
        h = math.sqrt(0.5)
        geoms.append(
            GeomSpec(
                shape=lobs.Capsule(radius=C.RIM_TUBE_R, length=seg),
                pose=Pose(
                    pos=(C.RIM_RING_R * c, C.RIM_RING_R * s, 0.0), quat=(h, -h * c, -h * s, 0.0)
                ),
            )
        )
    rim = w.add_actor("rim", root="fixed", pos=(0, 0, C.RIM_TUBE_Z))
    rim.add_body("ring", geoms=geoms)


def dyn_slots(sim) -> dict[str, int]:
    """Index of each moving body in ``sim.state()`` arrays: 'ball' and every player's 'body'."""
    out = {}
    for info in sim.reg.bodies:
        if info.path == "ball/body":
            out["ball"] = int(info.dyn)
        elif info.path.endswith("/body") and info.path.split("/")[0] in PLAYERS:
            out[info.path.split("/")[0]] = int(info.dyn)
    return out


def build_world(mode: str = "train") -> lobs.World:
    if mode not in ("train", "show"):
        raise ValueError(f"mode must be 'train' or 'show', got {mode!r}")
    w = lobs.World(f"lobpass-{mode}", dt=C.DT, substeps=C.SUBSTEPS)
    w.spec.backend_options.mujoco.solref_timeconst = C.CONTACT_TIMECONST
    w.add_plane("floor", friction=C.FLOOR_FRICTION)
    w.add_box(
        "backboard", C.BOARD_SIZE, fixed=True, pos=C.BOARD_CENTER, restitution=C.BOARD_RESTITUTION
    )
    w.add_box("bracket", C.BRACKET_SIZE, fixed=True, pos=C.BRACKET_CENTER)
    _rim(w)
    w.add_sphere("ball", C.BALL_R, mass=C.BALL_MASS, pos=(5.0, 0.0, 2.0), restitution=C.BALL_RESTITUTION,
                 friction=C.BALL_FRICTION, collision_group=C.GROUP_BALL)  # fmt: skip
    for i, name in enumerate(PLAYERS):
        if name in DEFENDERS:
            _rig(w, name, C.PARK_X + 3.0 * DEFENDERS.index(name), C.PARK_Y)
        else:
            _rig(w, name, 2.0 * i + 4.0, 0.0)
    pairs = w.spec.contact.disabled_group_pairs
    for p in [(C.GROUP_BALL, C.GROUP_PLAYER), (C.GROUP_PLAYER, C.GROUP_WORLD)] + [
        (C.GROUP_RIG, g) for g in (C.GROUP_WORLD, C.GROUP_BALL, C.GROUP_PLAYER, C.GROUP_RIG)
    ]:
        pairs.append(p)

    # ---- the passer's machine only carries the hold offset (A3); it never acts
    sm = lobs.StateMachine("passer")
    sm.var("hold_offset", (0.0, 0.0, 1.25 * C.PASSER_H_S0))
    sm.state("idle")
    w.add_behavior("passer", on="passer", spec=sm, period=C.DUNKER_PERIOD)

    # ---- ball forces: quadratic drag and the hold (A3)
    speed = dimless(props.speed("ball"))
    w.add_force("drag", on="ball", vector=newton(ball_vel() * speed * -C.DRAG_C))
    released = mhelp.released()
    flat = ops.vec(1, 1, 0)
    hold = pos_of("passer") * flat + dimless(props.behavior_var("passer", "hold_offset"))
    grip = (hold - ball_pos()) * C.HOLD_K - ball_vel() * C.HOLD_D + ops.vec(0, 0, C.BALL_MASS * C.G)
    w.add_force("hold", on="ball", vector=newton(grip), when=~released)
    w.measure("released", released)
    from . import (
        machines,  # local import: machines imports mhelp, not world
        rules,
    )

    r = rules.build(w)
    machines.build_dunker(w, blocked=r.blocked)
    for i in range(C.N_DEF_SLOTS):
        machines.build_defender(w, i)
    w.measure("dunker_state", props.behavior_state("dunker"))
    return w


def to_reset_args(scn) -> tuple[dict, dict]:
    """(initial_state, randomize) for ``sim.reset``: poses through ``initial_state`` and every machine variable
    through ``randomize['behaviors']`` (constants pass through unchanged, so nothing is random here)."""
    px, py = scn.passer_xy
    to_rim = -np.array([px, py]) / max(math.hypot(px, py), 1e-9)
    hold_off = (C.HOLD_FWD * to_rim[0], C.HOLD_FWD * to_rim[1], C.HOLD_Z_FRAC * scn.passer_H)
    init = {
        "ball": lobs.Initial(
            pos=(px + hold_off[0], py + hold_off[1], hold_off[2]), linvel=(0, 0, 0)
        ),
        "passer": lobs.Initial(q={"jx": px, "jy": py}),
        "dunker": lobs.Initial(q={"jx": scn.dunker_start[0], "jy": scn.dunker_start[1]}),
    }
    beh = {
        "passer": {"variables": {"hold_offset": hold_off}},
        "dunker": {"variables": {
            "corner": (*(scn.corner or (0.0, 0.0)), 0.0), "takeoff": (*scn.takeoff, 0.0),
            "has_corner": 1.0 if scn.corner is not None else 0.0, "vmax": scn.vmax, "v_corner": scn.v_corner,
            "v_gather": scn.v_gather, "start_delay": scn.start_delay, "jump_vz": scn.jump_vz,
            "jump_vh": scn.jump_vh, "H": scn.dunker_H, "R_s": scn.dunker_R_s, "L": scn.dunker_L, "present": 1.0,
        }},
    }  # fmt: skip
    live = {d.slot: d for d in scn.defenders}
    for i, name in enumerate(DEFENDERS):
        d = live.get(i)
        if d is None:
            init[name] = lobs.Initial(q={"jx": C.PARK_X + 3.0 * i, "jy": C.PARK_Y})
            beh[name] = {"variables": {"present": 0.0, "role": 0.0}}
            continue
        init[name] = lobs.Initial(q={"jx": d.start[0], "jy": d.start[1]})
        beh[name] = {"variables": {
            "present": 1.0, "role": float(d.role), "home": (d.start[0], d.start[1], 0.0), "speed": d.speed,
            "R_s": d.R_s, "L": d.L, "vz0": jump_cmd_vz(d.L), "react": d.react, "jump_ready": 1.0,
        }}  # fmt: skip
    return init, {"behaviors": beh}
