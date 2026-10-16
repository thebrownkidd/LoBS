"""Property and relative-measure implementations (plan sections 8.2 and 8.3)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .. import math3d as m3
from ..backends.capabilities import Capability
from ..errors import MeasureError

NAN = float("nan")
_EZ = np.array([0.0, 0.0, 1.0])


@dataclass
class PropDef:
    name: str
    targets: tuple[str, ...]
    shape: tuple
    unit: object  # str, or callable(reg, thing, args) -> str
    fn: Callable  # fn(ctx, thing, args, state) -> value
    needs: frozenset = frozenset({"pose"})
    kind: str = "real"
    prov: str = "computed"
    cap: Capability | None = None
    stateful: bool = False
    arg_names: tuple = ()
    doc: str = ""


PROPS: dict[str, PropDef] = {}


def prop(name, targets, shape, unit, needs=("pose",), kind="real", prov="computed", cap=None,
         stateful=False, args=(), doc=""):  # fmt: skip
    def deco(fn):
        PROPS[name] = PropDef(name, tuple(targets), tuple(shape), unit, fn, frozenset(needs), kind,
                              prov, cap, stateful, tuple(args), doc)  # fmt: skip
        return fn

    return deco


_FRAMES = ("actor", "body", "anchor")


# ---------------------------------------------------------------- geometry
@prop("position", _FRAMES, (3,), "meter")
def _position(ctx, th, a, s):
    return ctx.pose(th)[0]


@prop("height", _FRAMES, (), "meter")
def _height(ctx, th, a, s):
    return ctx.pose(th)[0][2]


@prop("orientation", _FRAMES, (4,), "dimensionless", kind="quat")
def _orientation(ctx, th, a, s):
    q = ctx.pose(th)[2]
    return -q if q[0] < 0 else q


@prop("tilt", _FRAMES, (), "radian")
def _tilt(ctx, th, a, s):
    return m3.angle_between(ctx.pose(th)[1][:, 2], _EZ)


@prop("heading", _FRAMES, (), "radian")
def _heading(ctx, th, a, s):
    x = ctx.pose(th)[1][:, 0]
    return NAN if np.hypot(x[0], x[1]) < 1e-12 else np.arctan2(x[1], x[0])


# ---------------------------------------------------------------- motion
@prop("velocity", _FRAMES, (3,), "meter/second", needs=("twist",), prov="engine")
def _velocity(ctx, th, a, s):
    return ctx.twist(th)[0]


@prop("speed", _FRAMES, (), "meter/second", needs=("twist",))
def _speed(ctx, th, a, s):
    return np.linalg.norm(ctx.twist(th)[0])


@prop("angular_velocity", _FRAMES, (3,), "radian/second", needs=("twist",), prov="engine")
def _angvel(ctx, th, a, s):
    return ctx.twist(th)[1]


@prop("spin", _FRAMES, (), "radian/second", needs=("twist",))
def _spin(ctx, th, a, s):
    return np.linalg.norm(ctx.twist(th)[1])


@prop("acceleration", ("actor", "body"), (3,), "meter/second**2", needs=("accel", "pose"),
      prov="engine", cap=Capability.BODY_ACCELERATION)  # fmt: skip
def _acceleration(ctx, th, a, s):
    f = ctx.frame
    if f.body_linacc is None:
        raise MeasureError("acceleration requires acceleration data in the state frame")

    def one(b):
        info = ctx.reg.bodies[b]
        if info.static:
            return np.zeros(3)
        _, _, r = ctx.body_pose(b)
        rc = r @ info.com
        w = f.body_angvel[info.dyn]
        return (
            f.body_linacc[info.dyn]
            + np.cross(f.body_angacc[info.dyn], rc)
            + np.cross(w, np.cross(w, rc))
        )

    if th.kind == "body":
        return (
            one(th.index)
            if th.point == "com"
            else (
                np.zeros(3)
                if ctx.reg.bodies[th.index].static
                else f.body_linacc[ctx.reg.bodies[th.index].dyn]
            )
        )
    actor = ctx.reg.actors[th.index]
    if actor.mass <= 0:
        return np.zeros(3)
    return sum(ctx.reg.bodies[b].mass * one(b) for b in actor.bodies) / actor.mass


@prop("displacement", _FRAMES, (3,), "meter", stateful=True)
def _displacement(ctx, th, a, s):
    p = ctx.pose(th)[0]
    if "p0" not in s:
        s["p0"] = p.copy()
    return p - s["p0"]


@prop("path_length", _FRAMES, (), "meter", needs=("twist",), prov="estimated", stateful=True)
def _path_length(ctx, th, a, s):
    sp = np.linalg.norm(ctx.twist(th)[0])
    if "t" not in s:
        s["len"] = 0.0
    else:
        s["len"] += 0.5 * (sp + s["sp"]) * (ctx.t - s["t"])
    s["t"], s["sp"] = ctx.t, sp
    return s["len"]


# ---------------------------------------------------------------- energy & momentum
def _bodies(ctx, th):
    return ctx.reg.dyn_bodies_of(th)


@prop("kinetic_energy", ("actor", "body"), (), "joule", needs=("twist", "mass"))
def _ke(ctx, th, a, s):
    total = 0.0
    for b in _bodies(ctx, th):
        info = ctx.reg.bodies[b]
        if info.static or info.mass <= 0:
            continue
        _, _, r = ctx.body_pose(b)
        _, w = ctx.body_twist(b)
        vc = ctx.body_com(b)[1]
        total += 0.5 * info.mass * vc @ vc + 0.5 * w @ (r @ info.inertia @ r.T) @ w
    return total


@prop("potential_energy", ("actor", "body"), (), "joule", needs=("pose", "mass"))
def _pe(ctx, th, a, s):
    g = ctx.reg.gravity
    return sum(-ctx.reg.bodies[b].mass * (g @ ctx.body_com(b)[0]) for b in _bodies(ctx, th))


@prop("mechanical_energy", ("actor", "body"), (), "joule", needs=("twist", "mass"))
def _me(ctx, th, a, s):
    return _ke(ctx, th, a, s) + _pe(ctx, th, a, s)


@prop("momentum", ("actor", "body"), (3,), "kilogram*meter/second", needs=("twist", "mass"))
def _momentum(ctx, th, a, s):
    return sum((ctx.reg.bodies[b].mass * ctx.body_com(b)[1] for b in _bodies(ctx, th)), np.zeros(3))


@prop(
    "angular_momentum", ("actor", "body"), (3,), "kilogram*meter**2/second", needs=("twist", "mass")
)
def _angmom(ctx, th, a, s):
    bodies = _bodies(ctx, th)
    mass = sum(ctx.reg.bodies[b].mass for b in bodies)
    if mass <= 0:
        return np.zeros(3)
    centre = sum(ctx.reg.bodies[b].mass * ctx.body_com(b)[0] for b in bodies) / mass
    total = np.zeros(3)
    for b in bodies:
        info = ctx.reg.bodies[b]
        if info.static or info.mass <= 0:
            continue
        pc, vc = ctx.body_com(b)
        _, _, r = ctx.body_pose(b)
        _, w = ctx.body_twist(b)
        total += np.cross(pc - centre, info.mass * vc) + (r @ info.inertia @ r.T) @ w
    return total


@prop("mass", ("actor", "body"), (), "kilogram", needs=("mass",), prov="model")
def _mass(ctx, th, a, s):
    return sum(ctx.reg.bodies[b].mass for b in _bodies(ctx, th))


# ---------------------------------------------------------------- contacts
def _body_set(ctx, th):
    return set(_bodies(ctx, th))


def _contact_mask(ctx, th, a):
    c = ctx.frame.contacts
    if c is None:
        raise MeasureError(
            "contact measures need contact data (record_contacts / contacts backend)"
        )
    mine = _body_set(ctx, th)
    other = a.get("_with")
    ia, ib = np.isin(c.body_a, list(mine)), np.isin(c.body_b, list(mine))
    if other is None:
        return c, ia | ib
    theirs = list(_body_set(ctx, other))
    return c, (ia & np.isin(c.body_b, theirs)) | (ib & np.isin(c.body_a, theirs))


@prop("in_contact", ("actor", "body"), (), "dimensionless", needs=("contacts",), kind="bool",
      prov="engine", cap=Capability.CONTACTS, args=("with",))  # fmt: skip
def _in_contact(ctx, th, a, s):
    return bool(np.any(_contact_mask(ctx, th, a)[1]))


@prop("contact_count", ("actor", "body"), (), "dimensionless", needs=("contacts",), prov="engine",
      cap=Capability.CONTACTS, args=("with",))  # fmt: skip
def _contact_count(ctx, th, a, s):
    return float(np.sum(_contact_mask(ctx, th, a)[1]))


@prop("contact_force", ("actor", "body"), (), "newton", needs=("contacts",), prov="engine",
      cap=Capability.CONTACT_FORCES, args=("with",))  # fmt: skip
def _contact_force(ctx, th, a, s):
    c, mask = _contact_mask(ctx, th, a)
    return float(np.sum(c.normal_force[mask]))


# ---------------------------------------------------------------- joints
def _joint_unit(rate: str):
    def unit(reg, th, args):
        j = reg.joints[th.index]
        base = "radian" if j.type == "hinge" else "meter"
        return base + rate

    return unit


def _dof(ctx, th):
    j = ctx.reg.joints[th.index]
    if j.dof < 0:
        raise MeasureError(f"joint '{j.path}' is fixed and has no state")
    return j.dof


@prop("joint_position", ("joint",), (), _joint_unit(""), needs=("joints",), prov="engine")
def _jq(ctx, th, a, s):
    return ctx.frame.joint_q[_dof(ctx, th)]


@prop("joint_velocity", ("joint",), (), _joint_unit("/second"), needs=("joints",), prov="engine")
def _jqd(ctx, th, a, s):
    return ctx.frame.joint_qd[_dof(ctx, th)]


@prop("joint_effort", ("joint",), (), lambda reg, th, a: "newton*meter" if reg.joints[th.index].type == "hinge" else "newton",
      needs=("joints",), prov="engine")  # fmt: skip
def _jeff(ctx, th, a, s):
    return ctx.frame.joint_effort[_dof(ctx, th)]


@prop("joint_limit_margin", ("joint",), (), _joint_unit(""), needs=("joints",))
def _jmargin(ctx, th, a, s):
    j = ctx.reg.joints[th.index]
    if j.range is None:
        raise MeasureError(f"joint '{j.path}' has no range")
    q = ctx.frame.joint_q[_dof(ctx, th)]
    return min(q - j.range[0], j.range[1] - q)


# ---------------------------------------------------------------- relative (distance & angle)
_LETTER = {"x": 0, "y": 1, "z": 2}


def parse_axis(axis) -> tuple[np.ndarray, int | None]:
    """-> (local unit vector, letter index or None)"""
    if isinstance(axis, str):
        sign = -1.0 if axis.startswith("-") else 1.0
        key = axis.lstrip("+-")
        if key not in _LETTER:
            raise MeasureError(f"axis must be x, y, z, -x, -y, -z or a 3-vector, got '{axis}'")
        v = np.zeros(3)
        v[_LETTER[key]] = sign
        return v, _LETTER[key]
    v = np.asarray(axis, dtype=float)
    if v.shape != (3,) or np.linalg.norm(v) < 1e-12:
        raise MeasureError("axis vector must be a non-zero 3-vector")
    return v / np.linalg.norm(v), None


def _perp_reference(rot: np.ndarray, u_local: np.ndarray, letter: int | None, u_w: np.ndarray):
    """Zero-direction for azimuth: the frame's next cyclic axis (letters) or least-aligned axis."""
    if letter is not None:
        ref = rot[:, (letter + 1) % 3]
    else:
        k = int(np.argmin(np.abs(u_local)))
        ref = rot[:, k]
    ref = ref - (ref @ u_w) * u_w
    return ref / np.linalg.norm(ref)


def eval_distance(ctx, things):
    a, b = ctx.pose(things[0])[0], ctx.pose(things[1])[0]
    return np.linalg.norm(b - a)


def eval_angle_pair(ctx, things, node, frame_thing):
    kind = node.angle_kind
    pa, ra, qa = ctx.pose(things[0])
    pb, rb, qb = ctx.pose(things[1])
    if kind == "rotation":
        return m3.quat_angle(m3.quat_mul(m3.quat_conj(qa), qb))
    if kind == "axes":
        ua, _ = parse_axis(node.axis)
        ub, _ = parse_axis(node.axis_b if node.axis_b is not None else node.axis)
        return m3.angle_between(ra @ ua, rb @ ub)
    d = pb - pa
    if np.linalg.norm(d) < 1e-12:
        return NAN
    rot = np.eye(3) if node.frame == "world" else (rb if node.frame == "b" else
          (ctx.pose(frame_thing)[1] if frame_thing is not None else ra))  # fmt: skip
    u_local, letter = parse_axis(node.axis)
    u = rot @ u_local
    polar = m3.angle_between(d, u)
    if kind == "polar":
        return polar
    if kind == "elevation":
        return np.pi / 2 - polar
    if kind == "azimuth":
        dp = d - (d @ u) * u
        if np.linalg.norm(dp) < 1e-12:
            return NAN
        ref = _perp_reference(rot, u_local, letter, u)
        return np.arctan2(u @ np.cross(ref, dp), ref @ dp)
    raise MeasureError(f"unknown angle kind '{kind}'")


def eval_angle_triple(ctx, things, node, frame_thing):
    pa, pv, pb = (ctx.pose(t)[0] for t in (things[0], things[1], things[2]))
    a, b = pa - pv, pb - pv
    if np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return NAN
    if node.signed_about is None:
        return m3.angle_between(a, b)
    rot = np.eye(3) if node.frame == "world" else (
        ctx.pose(frame_thing)[1] if frame_thing is not None else ctx.pose(things[1])[1]
    )  # fmt: skip
    u = rot @ parse_axis(node.signed_about)[0]
    return np.arctan2(u @ np.cross(a, b), a @ b)
