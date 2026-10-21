"""The one LoBS world: court, hoop, ball (drag + hold), seven player rigs, passer machine.

Training and evaluation use mode="train" and must stay on the native tick core (see tests/lobpass/test_world.py).
Every player is a fixed-root actor on x/y/z slide joints; per-episode heights live in behavior variables only.
"""

from __future__ import annotations

import math

import lobs
from lobs import ops, props
from lobs.schema import GeomSpec, MaterialSpec, Pose

from . import constants as C

PLAYERS = ("passer", "dunker", *[f"d{i}" for i in range(C.N_DEF_SLOTS)])
DEFENDERS = PLAYERS[2:]


def dimless(m):
    return ops.as_unit(m, "dimensionless")


def newton(m):
    return ops.as_unit(m, "newton")


def body_path(player: str) -> str:
    return f"{player}/body"


def pos_of(player: str):
    """Player position as a unit-free 3-vector measure: (x, y, jz), the body origin at floor level."""
    return dimless(props.position(f"{body_path(player)}:origin"))


def vel_of(player: str):
    return dimless(props.velocity(f"{body_path(player)}:origin"))


def ball_pos():
    return dimless(props.position("ball"))


def ball_vel():
    return dimless(props.velocity("ball"))


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
    released = ops.latch(speed > C.RELEASE_SPEED, ops.time() < -1.0)
    flat = ops.vec(1, 1, 0)
    hold = pos_of("passer") * flat + dimless(props.behavior_var("passer", "hold_offset"))
    grip = (hold - ball_pos()) * C.HOLD_K - ball_vel() * C.HOLD_D + ops.vec(0, 0, C.BALL_MASS * C.G)
    w.add_force("hold", on="ball", vector=newton(grip), when=~released)
    w.measure("released", released)
    return w
