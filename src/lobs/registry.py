"""Compiled, immutable index of a WorldSpec: global body order, static/dynamic split,
mass properties, joints, actuators, anchors and path resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import math3d as m3
from .errors import LobsError
from .inertia import resolve_body_inertial
from .schema import AnchorSpec, WorldSpec


@dataclass
class BodyInfo:
    path: str
    actor: int
    name: str
    parent: int  # global index, -1 for actor roots
    joint: int  # global joint index attaching this body to its parent, -1 for roots
    rel_pos: np.ndarray
    rel_quat: np.ndarray
    mass: float
    com: np.ndarray
    inertia: np.ndarray
    collision_group: int
    static: bool = False
    dyn: int = -1  # position in the dynamic-body arrays, -1 if static
    world_pos: np.ndarray | None = None  # static bodies only
    world_quat: np.ndarray | None = None


@dataclass
class JointInfo:
    path: str
    name: str
    actor: int
    type: str
    parent: int
    child: int
    pos: np.ndarray
    axis: np.ndarray  # unit
    range: tuple[float, float] | None
    damping: float
    armature: float
    dof: int = -1  # first index in joint_q arrays, -1 for fixed
    ndof: int = 0  # 1 for hinge/slide, 3 for ball (rotation vector)


@dataclass
class ActuatorInfo:
    path: str
    name: str
    joint: int
    kind: str
    kp: float
    kd: float
    ctrl_range: tuple[float, float] | None
    effort_limit: float | None


@dataclass
class AnchorInfo:
    path: str
    parent: int  # global body index or -1 for world
    pos: np.ndarray
    quat: np.ndarray
    rule: object
    tags: list[str]


@dataclass
class ActorInfo:
    path: str
    root_free: bool
    bodies: list[int]
    tags: list[str]
    mass: float = 0.0


@dataclass
class Thing:
    """A resolved reference: world | actor | body | anchor | joint."""

    kind: str
    index: int
    point: str = "com"  # 'com' or 'origin' (actors/bodies)
    path: str = ""


@dataclass
class Registry:
    spec: WorldSpec
    bodies: list[BodyInfo] = field(default_factory=list)
    joints: list[JointInfo] = field(default_factory=list)
    actuators: list[ActuatorInfo] = field(default_factory=list)
    anchors: list[AnchorInfo] = field(default_factory=list)
    actors: list[ActorInfo] = field(default_factory=list)
    dynamic_order: list[int] = field(default_factory=list)
    dof_joints: list[int] = field(default_factory=list)
    n_dof: int = 0  # length of joint_q / joint_qd / joint_effort
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))
    base_dir: object = None  # directory that relative asset paths (meshes) resolve against
    _paths: dict = field(default_factory=dict)

    @property
    def body_paths(self) -> list[str]:
        return [b.path for b in self.bodies]

    # ------------------------------------------------------------------ resolve
    def resolve(self, ref) -> Thing:
        """Resolve a path string (optionally suffixed ':com' / ':origin') to a Thing."""
        path = str(getattr(ref, "path", ref))
        point = None
        if ":" in path:
            path, point = path.rsplit(":", 1)
            if point not in ("com", "origin"):
                raise LobsError(f"unknown frame suffix ':{point}' (use ':com' or ':origin')")
        if path == "world":
            if point == "com":
                raise LobsError("the world has no ':com'")
            return Thing("world", -1, "origin", "world")
        hit = self._paths.get(path)
        if hit is None:
            raise LobsError(f"unknown path '{path}'")
        kind, idx = hit
        if kind in ("anchor", "joint") and point is not None:
            raise LobsError(f"'{path}' is a {kind}; ':{point}' does not apply")
        return Thing(kind, idx, point or "com", path)

    def dyn_bodies_of(self, thing: Thing) -> list[int]:
        if thing.kind == "actor":
            return list(self.actors[thing.index].bodies)
        if thing.kind == "body":
            return [thing.index]
        raise LobsError(f"'{thing.path}' is a {thing.kind}, not an actor or body")

    def resolve_actuator(self, key: str) -> int:
        for i, a in enumerate(self.actuators):
            if a.path == key:
                return i
        hits = [i for i, a in enumerate(self.actuators) if a.name == key]
        if len(hits) == 1:
            return hits[0]
        raise LobsError(
            f"unknown actuator '{key}'" if not hits else f"ambiguous actuator '{key}': {len(hits)}"
        )


def _check_unique(names: list[str], what: str, actor: str) -> None:
    seen = set()
    for n in names:
        if n in seen:
            raise LobsError(f"actor '{actor}': duplicate {what} name '{n}'")
        seen.add(n)


def build_registry(spec: WorldSpec, base_dir=None) -> Registry:
    reg = Registry(spec=spec, gravity=np.array(spec.gravity, dtype=float), base_dir=base_dir)
    default_density = spec.contact.default_material.density or 1000.0
    paths = reg._paths

    def claim(path: str, kind: str, idx: int):
        if path in paths:
            raise LobsError(f"duplicate path '{path}'")
        paths[path] = (kind, idx)

    actor_names = [a.name for a in spec.actors]
    _check_unique(actor_names, "actor", "<world>")
    if "world" in actor_names:
        raise LobsError("'world' is reserved and cannot be an actor name")

    for ai, actor in enumerate(spec.actors):
        _check_unique(
            [b.name for b in actor.bodies]
            + [j.name for j in actor.joints]
            + [a.name for a in actor.anchors]
            + [a.name for a in actor.actuators],
            "body/joint/anchor/actuator",
            actor.name,
        )
        if not actor.bodies:
            raise LobsError(f"actor '{actor.name}' has no bodies")
        claim(actor.name, "actor", ai)
        local = {}
        ainfo = ActorInfo(actor.name, actor.root == "free", [], list(actor.tags))
        reg.actors.append(ainfo)
        joint_by_child = {}
        for j in actor.joints:
            if j.child is None or j.parent is None:
                raise LobsError(f"joint '{actor.name}/{j.name}' needs parent and child")
            if j.child in joint_by_child:
                raise LobsError(f"body '{actor.name}/{j.child}' has more than one joint")
            joint_by_child[j.child] = j
        for bi, b in enumerate(actor.bodies):
            gi = len(reg.bodies)
            if b.parent is None:
                if bi != 0:
                    raise LobsError(
                        f"body '{actor.name}/{b.name}': only the first body may be a root"
                    )
                parent, joint = -1, -1
            else:
                if b.parent not in local:
                    raise LobsError(
                        f"body '{actor.name}/{b.name}': unknown/late parent '{b.parent}'"
                    )
                parent = local[b.parent]
                if b.name not in joint_by_child:
                    raise LobsError(f"body '{actor.name}/{b.name}' has no joint to its parent")
                joint = -2  # patched below
            if b.parent is None and b.name in joint_by_child:
                raise LobsError(f"root body '{actor.name}/{b.name}' cannot have a joint")
            for g in b.geoms:
                if g.shape.type == "plane" and (actor.root != "fixed" or b.parent is not None):
                    raise LobsError(f"plane geoms are only allowed on fixed roots ('{actor.name}')")
            mass, com, inertia = resolve_body_inertial(b, default_density)
            info = BodyInfo(
                path=f"{actor.name}/{b.name}",
                actor=ai,
                name=b.name,
                parent=parent,
                joint=joint,
                rel_pos=np.array(b.pose.pos),
                rel_quat=np.array(b.pose.quat),
                mass=mass,
                com=com,
                inertia=inertia,
                collision_group=b.collision_group,
            )
            local[b.name] = gi
            reg.bodies.append(info)
            ainfo.bodies.append(gi)
            claim(info.path, "body", gi)
        for j in actor.joints:
            if j.child not in local or j.parent not in local:
                raise LobsError(f"joint '{actor.name}/{j.name}': unknown parent/child body")
            if reg.bodies[local[j.child]].parent != local[j.parent]:
                raise LobsError(f"joint '{actor.name}/{j.name}' parent is not the body's parent")
            axis = np.array(j.axis, dtype=float)
            if j.type in ("hinge", "slide") and np.linalg.norm(axis) < 1e-12:
                raise LobsError(f"joint '{actor.name}/{j.name}' has a zero axis")
            ji = JointInfo(
                path=f"{actor.name}/{j.name}",
                name=j.name,
                actor=ai,
                type=j.type,
                parent=local[j.parent],
                child=local[j.child],
                pos=np.array(j.pos),
                axis=axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis,
                range=j.range,
                damping=j.damping,
                armature=j.armature,
            )
            reg.bodies[ji.child].joint = len(reg.joints)
            if j.type != "fixed":
                ji.dof = reg.n_dof
                ji.ndof = 3 if j.type == "ball" else 1
                reg.n_dof += ji.ndof
                reg.dof_joints.append(len(reg.joints))
            claim(ji.path, "joint", len(reg.joints))
            reg.joints.append(ji)
        jlocal = {j.name: i for i, j in enumerate(reg.joints) if j.actor == ai}
        for act in actor.actuators:
            if act.joint not in jlocal or reg.joints[jlocal[act.joint]].type not in (
                "hinge",
                "slide",
            ):
                raise LobsError(
                    f"actuator '{actor.name}/{act.name}': joint '{act.joint}' has no DOF"
                )
            reg.actuators.append(
                ActuatorInfo(
                    f"{actor.name}/{act.name}",
                    act.name,
                    jlocal[act.joint],
                    act.kind,
                    act.kp,
                    act.kd,
                    act.ctrl_range,
                    act.effort_limit,
                )  # fmt: skip
            )
        for a in actor.anchors:
            if a.parent not in local:
                raise LobsError(f"anchor '{actor.name}/{a.name}': unknown parent body '{a.parent}'")
            _add_anchor(reg, claim, f"{actor.name}/{a.name}", local[a.parent], a)
        ainfo.mass = sum(reg.bodies[i].mass for i in ainfo.bodies)

    for a in spec.anchors:
        if a.parent != "world":
            raise LobsError(f"world anchor '{a.name}' must have parent 'world'")
        _add_anchor(reg, claim, f"world/{a.name}", -1, a)

    _classify_static_and_poses(reg, spec)
    return reg


def _add_anchor(reg: Registry, claim, path: str, parent: int, a: AnchorSpec) -> None:
    claim(path, "anchor", len(reg.anchors))
    reg.anchors.append(
        AnchorInfo(path, parent, np.array(a.pose.pos), np.array(a.pose.quat), a.rule, list(a.tags))
    )


def _classify_static_and_poses(reg: Registry, spec: WorldSpec) -> None:
    from .kinematics import fk_actor

    for ai, ainfo in enumerate(reg.actors):
        actor = spec.actors[ai]
        for gi in ainfo.bodies:
            b = reg.bodies[gi]
            if b.parent < 0:
                b.static = actor.root == "fixed"
            else:
                b.static = reg.bodies[b.parent].static and reg.joints[b.joint].type == "fixed"
        if any(reg.bodies[gi].static for gi in ainfo.bodies):
            q0 = np.zeros(reg.n_dof)
            res = fk_actor(reg, ai, actor.pose, q0, q0, np.zeros(3), np.zeros(3))
            for gi, (p, q, _, _) in res.items():
                if reg.bodies[gi].static:
                    reg.bodies[gi].world_pos, reg.bodies[gi].world_quat = p, q
    for gi, b in enumerate(reg.bodies):
        if not b.static:
            b.dyn = len(reg.dynamic_order)
            reg.dynamic_order.append(gi)


@dataclass
class ResolvedInitial:
    """Per-actor effective pose/velocity and joint states, in registry order."""

    actor_pose: list  # [(pos, quat)] per actor
    linvel: np.ndarray  # (n_actors, 3)
    angvel: np.ndarray  # (n_actors, 3)
    q: np.ndarray  # (n_dof,)
    qd: np.ndarray


def resolve_initial(reg: Registry, overrides: dict | None = None) -> ResolvedInitial:
    """Merge the spec's InitialState with runtime overrides {actor_path: Initial-like}."""
    overrides = overrides or {}
    unknown = set(overrides) - {a.path for a in reg.actors}
    if unknown:
        raise LobsError(f"initial_state refers to unknown actors: {sorted(unknown)}")
    n = len(reg.actors)
    poses, lin, ang = [], np.zeros((n, 3)), np.zeros((n, 3))
    q, qd = np.zeros(reg.n_dof), np.zeros(reg.n_dof)
    for ai, actor in enumerate(reg.spec.actors):
        init = actor.initial
        ov = overrides.get(actor.name)
        pos, quat = np.array(actor.pose.pos), np.array(actor.pose.quat)
        if init.pose is not None:
            pos, quat = np.array(init.pose.pos), np.array(init.pose.quat)
        lv, av = np.array(init.linvel), np.array(init.angvel)
        qs, qds = dict(init.q), dict(init.qd)
        if ov is not None:
            if ov.pos is not None:
                pos = np.array(ov.pos, dtype=float)
            if ov.quat is not None:
                quat = m3.quat_normalize(np.array(ov.quat, dtype=float))
            if ov.linvel is not None:
                lv = np.array(ov.linvel, dtype=float)
            if ov.angvel is not None:
                av = np.array(ov.angvel, dtype=float)
            qs.update(ov.q or {})
            qds.update(ov.qd or {})
        poses.append((pos, quat))
        lin[ai], ang[ai] = lv, av
        for name, val in qs.items():
            _set_dof(reg, actor.name, name, val, q, "q")
        for name, val in qds.items():
            _set_dof(reg, actor.name, name, val, qd, "qd")
    return ResolvedInitial(poses, lin, ang, q, qd)


def _set_dof(reg: Registry, actor: str, joint: str, val, target: np.ndarray, what: str) -> None:
    hit = reg._paths.get(f"{actor}/{joint}")
    if hit is None or hit[0] != "joint" or reg.joints[hit[1]].dof < 0:
        raise LobsError(f"initial {what}: '{actor}/{joint}' is not a movable joint")
    j = reg.joints[hit[1]]
    arr = np.atleast_1d(np.asarray(val, dtype=float))
    if arr.shape != (j.ndof,):
        kind = "a 3-vector (rotation vector / angular velocity)" if j.ndof == 3 else "a number"
        raise LobsError(f"initial {what} for '{j.path}' ({j.type} joint) must be {kind}")
    target[j.dof : j.dof + j.ndof] = arr
