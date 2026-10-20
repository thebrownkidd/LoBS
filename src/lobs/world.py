"""World builder facade and handles. The pure data lives in ``lobs.schema.WorldSpec``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import faces
from . import math3d as m3
from .errors import LobsError
from .schema import (
    ActorSpec,
    AnchorSpec,
    BodySpec,
    Box,
    Capsule,
    Cylinder,
    GeomSpec,
    InertialSpec,
    JointSpec,
    MaterialSpec,
    Placement,
    Plane,
    Pose,
    Sphere,
    WorldSpec,
)
from .schema.actor import ActuatorSpec

WORLD = "world:origin"


# ------------------------------------------------------------------ joint constructors
def Hinge(name, axis=(0, 0, 1), pos=(0, 0, 0), range=None, damping=0.0, armature=0.0) -> JointSpec:  # noqa: A002
    return JointSpec(name=name, type="hinge", axis=tuple(axis), pos=tuple(pos),
                     range=range, damping=damping, armature=armature)  # fmt: skip


def Slide(name, axis=(1, 0, 0), pos=(0, 0, 0), range=None, damping=0.0, armature=0.0) -> JointSpec:  # noqa: A002
    return JointSpec(name=name, type="slide", axis=tuple(axis), pos=tuple(pos),
                     range=range, damping=damping, armature=armature)  # fmt: skip


def Fixed(name=None) -> JointSpec:
    return JointSpec(name=name or "", type="fixed")


@dataclass
class Initial:
    """Runtime initial-state override for one actor (None = keep the world's value)."""

    pos: tuple | None = None
    quat: tuple | None = None
    linvel: tuple | None = None
    angvel: tuple | None = None
    q: dict = field(default_factory=dict)
    qd: dict = field(default_factory=dict)


# ------------------------------------------------------------------ handles
class _Handle:
    def __init__(self, world: World, path: str):
        self._world = world
        self.path = path

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.path!r})"

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _Handle) and other.path == self.path and other._world is self._world
        )

    def __hash__(self) -> int:
        return hash(self.path)

    @property
    def com(self) -> str:
        return f"{self.path}:com"

    @property
    def origin(self) -> str:
        return f"{self.path}:origin"


class AnchorHandle(_Handle):
    @property
    def com(self):
        raise LobsError("anchors have no ':com'")

    origin = com


class JointHandle(_Handle):
    @property
    def com(self):
        raise LobsError("joints have no ':com'")

    origin = com


def _geoms(geoms) -> list[GeomSpec]:
    if geoms is None:
        return []
    if not isinstance(geoms, (list, tuple)):
        geoms = [geoms]
    return [g if isinstance(g, GeomSpec) else GeomSpec(shape=g) for g in geoms]


def _inertial(mass, density, inertial) -> InertialSpec:
    if inertial is not None:
        return inertial
    return InertialSpec(mode="auto", mass=mass, density=density)


class _AnchorMixin:
    """Anchor creation shared by actor and body handles."""

    def _actor_spec(self) -> ActorSpec:
        raise NotImplementedError

    def _default_body(self) -> str:
        raise NotImplementedError

    def _add_anchor(self, spec: AnchorSpec) -> AnchorHandle:
        a = self._actor_spec()
        if any(n == spec.name for n in _names(a)):
            raise LobsError(f"actor '{a.name}': name '{spec.name}' already used")
        a.anchors.append(spec)
        return AnchorHandle(self._world, f"{a.name}/{spec.name}")

    def anchor(self, name, pos=(0, 0, 0), quat=None, body=None, rule=None, tags=()) -> AnchorHandle:
        kw = {"rule": rule} if rule is not None else {}
        return self._add_anchor(
            AnchorSpec(
                name=name,
                parent=body or self._default_body(),
                pose=Pose(pos=tuple(pos), quat=tuple(quat) if quat is not None else (1, 0, 0, 0)),
                tags=list(tags),
                **kw,
            )  # fmt: skip
        )

    def anchor_on_face(self, face=None, name=None, body=None, uv=(0.0, 0.0), offset=0.0,
                       angle=0.0, direction=(0, 0, 1), rule=None, tags=()) -> AnchorHandle:  # fmt: skip
        a = self._actor_spec()
        bname = body or self._default_body()
        bspec = next((b for b in a.bodies if b.name == bname), None)
        if bspec is None:
            raise LobsError(f"actor '{a.name}' has no body '{bname}'")
        if not bspec.geoms:
            raise LobsError(f"body '{a.name}/{bname}' has no geometry to place an anchor on")
        g = next((x for x in bspec.geoms if x.role != "visual"), bspec.geoms[0])
        if g.shape.type in ("convex_mesh", "convex_set", "triangle_mesh"):
            from .assets import load_mesh

            files = [g.shape.file] if g.shape.type != "convex_set" else list(g.shape.files)
            parts = [load_mesh(self._world.base_dir, f) for f in files]
            verts = np.concatenate([p[0] for p in parts])
            offs = np.cumsum([0] + [len(p[0]) for p in parts[:-1]])
            tris = np.concatenate([p[1] + o for p, o in zip(parts, offs, strict=True)])
            use_dir = direction if (face is None or face not in faces._MESH_DIRS) else None
            pos, quat = faces.mesh_face_pose(verts, tris, face, use_dir, offset)
        else:
            pos, quat = faces.face_pose(g.shape, face, uv, offset, angle, direction)
        p, q = m3.pose_compose(g.pose.pos, g.pose.quat, pos, quat)
        kw = {"rule": rule} if rule is not None else {}
        spec = AnchorSpec(
            name=name or (face if face else "face"), parent=bname,
            pose=Pose(pos=tuple(float(x) for x in p), quat=tuple(float(x) for x in q)),
            placement=Placement(face=str(face), body=bname, uv=tuple(uv), offset=offset),
            tags=list(tags), **kw,
        )  # fmt: skip
        return self._add_anchor(spec)


def _names(a: ActorSpec) -> list[str]:
    return ([b.name for b in a.bodies] + [j.name for j in a.joints]
            + [x.name for x in a.anchors] + [x.name for x in a.actuators])  # fmt: skip


class BodyHandle(_Handle, _AnchorMixin):
    def __init__(self, world, actor: str, body: str):
        super().__init__(world, f"{actor}/{body}")
        self._actor, self._body = actor, body

    def _actor_spec(self):
        return self._world._actor(self._actor)

    def _default_body(self):
        return self._body

    def add_child(self, name, pos=(0, 0, 0), quat=None, joint=None, geoms=None, mass=None,
                  density=None, inertial=None, collision_group=0, tags=()) -> BodyHandle:  # fmt: skip
        a = self._actor_spec()
        if any(n == name for n in _names(a)):
            raise LobsError(f"actor '{a.name}': name '{name}' already used")
        joint = joint or Fixed()
        joint = joint.model_copy(update={"parent": self._body, "child": name})
        if not joint.name:
            joint = joint.model_copy(update={"name": f"{name}_weld"})
        if any(n == joint.name for n in _names(a)) or joint.name == name:
            raise LobsError(f"actor '{a.name}': name '{joint.name}' already used")
        a.bodies.append(
            BodySpec(
                name=name,
                parent=self._body,
                pose=Pose(pos=tuple(pos), quat=tuple(quat) if quat is not None else (1, 0, 0, 0)),
                geoms=_geoms(geoms),
                inertial=_inertial(mass, density, inertial),
                collision_group=collision_group,
                tags=list(tags),
            )  # fmt: skip
        )
        a.joints.append(joint)
        return BodyHandle(self._world, self._actor, name)


class ActorHandle(_Handle, _AnchorMixin):
    def _actor_spec(self):
        return self._world._actor(self.path)

    def _default_body(self):
        return self._actor_spec().bodies[0].name

    def add_body(self, name="base", pos=(0, 0, 0), quat=None, geoms=None, mass=None,
                 density=None, inertial=None, collision_group=0, tags=()) -> BodyHandle:  # fmt: skip
        a = self._actor_spec()
        if a.bodies:
            raise LobsError(f"actor '{a.name}' already has a root body; use body.add_child()")
        if any(n == name for n in _names(a)):
            raise LobsError(f"actor '{a.name}': name '{name}' already used")
        a.bodies.append(
            BodySpec(
                name=name,
                parent=None,
                pose=Pose(pos=tuple(pos), quat=tuple(quat) if quat is not None else (1, 0, 0, 0)),
                geoms=_geoms(geoms),
                inertial=_inertial(mass, density, inertial),
                collision_group=collision_group,
                tags=list(tags),
            )  # fmt: skip
        )
        return BodyHandle(self._world, a.name, name)

    def body(self, name=None) -> BodyHandle:
        a = self._actor_spec()
        name = name or a.bodies[0].name
        if not any(b.name == name for b in a.bodies):
            raise LobsError(f"actor '{a.name}' has no body '{name}'")
        return BodyHandle(self._world, a.name, name)

    def joint(self, name) -> JointHandle:
        a = self._actor_spec()
        if not any(j.name == name for j in a.joints):
            raise LobsError(f"actor '{a.name}' has no joint '{name}'")
        return JointHandle(self._world, f"{a.name}/{name}")

    def add_actuator(self, name, joint, kind="effort", kp=0.0, kd=0.0, ctrl_range=None,
                     effort_limit=None) -> str:  # fmt: skip
        a = self._actor_spec()
        if any(n == name for n in _names(a)):
            raise LobsError(f"actor '{a.name}': name '{name}' already used")
        jname = joint.path.split("/", 1)[1] if isinstance(joint, _Handle) else joint
        a.actuators.append(
            ActuatorSpec(
                name=name,
                joint=jname,
                kind=kind,
                kp=kp,
                kd=kd,
                ctrl_range=ctrl_range,
                effort_limit=effort_limit,
            )  # fmt: skip
        )
        return f"{a.name}/{name}"

    def set_initial(self, pos=None, quat=None, linvel=None, angvel=None, q=None, qd=None):
        a = self._actor_spec()
        if pos is not None or quat is not None:
            a.initial.pose = Pose(
                pos=tuple(pos) if pos is not None else a.pose.pos,
                quat=tuple(quat) if quat is not None else a.pose.quat,
            )
        if linvel is not None:
            a.initial.linvel = tuple(linvel)
        if angvel is not None:
            a.initial.angvel = tuple(angvel)
        a.initial.q.update(q or {})
        a.initial.qd.update(qd or {})
        return self


# ------------------------------------------------------------------ world
class World:
    def __init__(self, name="world", dt=0.002, substeps=1, duration=None, gravity=(0, 0, -9.81),
                 spec: WorldSpec | None = None, base_dir=None):  # fmt: skip
        if spec is not None:
            self.spec = spec
        else:
            self.spec = WorldSpec(name=name, gravity=tuple(gravity))
            self.spec.timing.dt, self.spec.timing.substeps = dt, substeps
            self.spec.timing.duration = duration
        self.base_dir = (
            None if base_dir is None else Path(base_dir)
        )  # where relative asset paths live

    # -- internals
    def _actor(self, name: str) -> ActorSpec:
        for a in self.spec.actors:
            if a.name == name:
                return a
        raise LobsError(f"unknown actor '{name}'")

    def _material(self, friction, restitution, density=None):
        if friction is None and restitution is None and density is None:
            return None
        return MaterialSpec(friction=friction, restitution=restitution, density=density)

    def add_actor(self, name, root="free", pos=(0, 0, 0), quat=None, tags=()) -> ActorHandle:
        if any(a.name == name for a in self.spec.actors) or name in ("world", "force", "behavior"):
            raise LobsError(f"actor name '{name}' already used or reserved")
        self.spec.actors.append(
            ActorSpec(
                name=name,
                root=root,
                pose=Pose(pos=tuple(pos), quat=tuple(quat) if quat is not None else (1, 0, 0, 0)),
                tags=list(tags),
            )  # fmt: skip
        )
        return ActorHandle(self, name)

    def _add_primitive(self, name, shape, mass, density, pos, quat, fixed, friction, restitution,
                       rgba, tags, linvel, angvel, collision_group):  # fmt: skip
        actor = self.add_actor(
            name, root="fixed" if fixed else "free", pos=pos, quat=quat, tags=tags
        )
        geom = GeomSpec(shape=shape, material=self._material(friction, restitution), rgba=rgba)
        actor.add_body(
            "body", geoms=[geom], mass=mass, density=density, collision_group=collision_group
        )
        if linvel is not None or angvel is not None:
            actor.set_initial(linvel=linvel, angvel=angvel)
        return actor

    def add_box(self, name, size, mass=None, density=None, pos=(0, 0, 0), quat=None, fixed=False,
                friction=None, restitution=None, rgba=None, tags=(), linvel=None, angvel=None,
                collision_group=0) -> ActorHandle:  # fmt: skip
        return self._add_primitive(name, Box(size=tuple(size)), mass, density, pos, quat, fixed,
                                   friction, restitution, rgba, tags, linvel, angvel, collision_group)  # fmt: skip

    def add_sphere(self, name, radius, mass=None, density=None, pos=(0, 0, 0), quat=None, fixed=False,
                   friction=None, restitution=None, rgba=None, tags=(), linvel=None, angvel=None,
                   collision_group=0) -> ActorHandle:  # fmt: skip
        return self._add_primitive(name, Sphere(radius=radius), mass, density, pos, quat, fixed,
                                   friction, restitution, rgba, tags, linvel, angvel, collision_group)  # fmt: skip

    def add_cylinder(self, name, radius, length, mass=None, density=None, pos=(0, 0, 0), quat=None,
                     fixed=False, friction=None, restitution=None, rgba=None, tags=(), linvel=None,
                     angvel=None, collision_group=0) -> ActorHandle:  # fmt: skip
        return self._add_primitive(name, Cylinder(radius=radius, length=length), mass, density, pos,
                                   quat, fixed, friction, restitution, rgba, tags, linvel, angvel,
                                   collision_group)  # fmt: skip

    def add_capsule(self, name, radius, length, mass=None, density=None, pos=(0, 0, 0), quat=None,
                    fixed=False, friction=None, restitution=None, rgba=None, tags=(), linvel=None,
                    angvel=None, collision_group=0) -> ActorHandle:  # fmt: skip
        return self._add_primitive(name, Capsule(radius=radius, length=length), mass, density, pos,
                                   quat, fixed, friction, restitution, rgba, tags, linvel, angvel,
                                   collision_group)  # fmt: skip

    def add_plane(self, name="ground", friction=None, restitution=None, pos=(0, 0, 0), quat=None,
                  rgba=None, tags=()) -> ActorHandle:  # fmt: skip
        return self._add_primitive(name, Plane(), None, None, pos, quat, True, friction, restitution,
                                   rgba, tags, None, None, 0)  # fmt: skip

    def anchor(self, name, pos=(0, 0, 0), quat=None, rule=None, tags=()) -> AnchorHandle:
        if any(a.name == name for a in self.spec.anchors):
            raise LobsError(f"world anchor '{name}' already exists")
        kw = {"rule": rule} if rule is not None else {}
        self.spec.anchors.append(
            AnchorSpec(
                name=name,
                parent="world",
                pose=Pose(pos=tuple(pos), quat=tuple(quat) if quat is not None else (1, 0, 0, 0)),
                tags=list(tags),
                **kw,
            )  # fmt: skip
        )
        return AnchorHandle(self, f"world/{name}")

    def actor(self, name) -> ActorHandle:
        self._actor(name)
        return ActorHandle(self, name)

    def find(self, tag=None, source=None, cls=None) -> list[ActorHandle]:
        out = []
        for a in self.spec.actors:
            if tag is not None and tag not in a.tags:
                continue
            if source is not None and not (a.source and a.source.source_id == source):
                continue
            if cls is not None and not (a.source and a.source.semantic_class == cls):
                continue
            out.append(ActorHandle(self, a.name))
        return out

    # -- force laws
    def add_force(
        self, name, on, magnitude=None, *, toward=None, direction=None, axis=None, vector=None,
        when=None, budget=None, adversary=False, reaction=True, tags=(),
    ):  # fmt: skip
        """Add a force law, evaluated every tick (zero-order hold over the substeps).

        Either ``magnitude`` (scalar Measure or number) along the unit vector ``toward`` another thing, a
        world ``direction``, or a frame ``axis=(thing, 'z')``; or a ``vector`` Measure (shape 3, newtons).
        ``when`` (bool Measure) gates it, ``budget=lobs.Budget(peak=, impulse=, active_time=)`` bounds it,
        ``adversary=True`` tags it. With ``toward=`` and ``reaction=True`` the equal and opposite force
        acts on ``toward``."""
        from .measure.nodes import as_measure
        from .schema.world import ForceBudget, ForceSpec

        if any(f.name == name for f in self.spec.forces):
            raise LobsError(f"force '{name}' already exists")
        path = lambda x: str(getattr(x, "path", x))  # noqa: E731
        if budget is not None and not isinstance(budget, ForceBudget):
            budget = ForceBudget(**budget)
        self.spec.forces.append(
            ForceSpec(
                name=name,
                on=path(on),
                magnitude=None if magnitude is None else as_measure(magnitude).node,
                toward=None if toward is None else path(toward),
                direction=None if direction is None else tuple(float(x) for x in direction),
                axis=None if axis is None else (path(axis[0]), str(axis[1])),
                vector=None if vector is None else as_measure(vector).node,
                when=None if when is None else as_measure(when).node,
                budget=budget,
                adversary=adversary,
                reaction=reaction,
                tags=list(tags),
            )  # fmt: skip
        )
        return self

    def add_behavior(self, name, on=None, spec=None, *, script=None, params=None, actuates=(), latency=0.0,
                     period=0.0, enabled=True, tags=()):  # fmt: skip
        """Add a behavior: a state machine (``spec=lobs.StateMachine(...)``) or a registered Python script
        (``script="name"`` from ``@lobs.behavior``), optionally for one actor (``on=``).

        ``latency`` (s) delays what it senses, ``period`` (s) sets how often it decides (commands are held
        in between). Actuators it commands are owned by it: external actions for them are ignored."""
        from .act import StateMachine
        from .behavior import policy_actuators
        from .schema.behavior import BehaviorSpec, ScriptSpec

        if any(b.name == name for b in self.spec.behaviors):
            raise LobsError(f"behavior '{name}' already exists")
        if (spec is None) == (script is None):
            raise LobsError("add_behavior: give exactly one of spec= (a state machine) or script=")
        path = lambda x: str(getattr(x, "path", x))  # noqa: E731
        if isinstance(spec, StateMachine):
            spec = spec.spec()
        machine = spec
        scr = None
        if script is not None:
            own = [path(a) for a in actuates] + [
                a for a in policy_actuators(script) if a not in map(path, actuates)
            ]
            scr = ScriptSpec(fn=script, params=dict(params or {}), actuates=own)
        self.spec.behaviors.append(
            BehaviorSpec(
                name=name,
                on=None if on is None else path(on),
                enabled=enabled,
                latency=latency,
                period=period,
                machine=machine,
                script=scr,
                tags=list(tags),
            )  # fmt: skip
        )
        return self

    def add_adversary(self, name, on, budget, **kwargs):
        """A force law that works against the agent. Same as ``add_force(..., adversary=True)`` but a
        ``budget`` with a ``peak`` is required, so adversaries are always bounded (and reproducible)."""
        from .schema.world import ForceBudget

        if not isinstance(budget, ForceBudget):
            budget = ForceBudget(**budget)
        if budget.peak is None:
            raise LobsError(
                f"adversary '{name}': budget.peak is required (adversaries must be bounded)"
            )
        return self.add_force(name, on, budget=budget, adversary=True, **kwargs)

    def add_disturbance(
        self, name, on, force, *, start: float = 0.0, duration: float | None = None
    ):
        """A constant world-frame force on ``on`` during [start, start + duration) (forever if no duration).
        Sugar over ``add_force``; use ``add_force`` with a time-dependent measure for other profiles."""
        from . import ops

        f = np.asarray(force, dtype=float)
        mag = float(np.linalg.norm(f))
        if mag == 0:
            raise LobsError("disturbance force must be non-zero")
        t = ops.time()
        active = (t >= start) if duration is None else ((t >= start) & (t < start + duration))
        return self.add_force(
            name, on, ops.where(active, mag, 0.0), direction=tuple(f / mag), reaction=False
        )

    # -- composition
    def include(self, other, prefix: str, pose=None):
        """Merge another world (or package path) into this one under ``prefix`` (names become
        ``<prefix>.<name>``). The host's gravity/timing win. Assets are copied into this world's
        asset directory; measures are renamed and their paths rewritten."""
        from .composition import include_world

        include_world(self, other, prefix, pose)
        return self

    # -- measures
    def measure(self, name, m):
        from .measure.nodes import Measure, as_measure

        if not name.replace(".", "_").replace("_", "a").isalnum() or name[0].isdigit():
            raise LobsError(f"invalid measure name '{name}'")
        if name in self.spec.measures:
            raise LobsError(f"measure '{name}' already exists")
        m = as_measure(m)
        self.spec.measures[name] = m.node
        from .schema.measure import RefNode

        return Measure(RefNode(name=name))

    # -- compile / run
    def registry(self):
        from .registry import build_registry

        return build_registry(self.spec, self.base_dir)

    def validate(self, backend=None):
        from .validate import validate_world

        return validate_world(self, backend)

    def simulate(self, backend="mujoco", **kw):
        from .sim import Simulation

        return Simulation.create(self, backend, **kw)

    def copy(self) -> World:
        return World(spec=self.spec.model_copy(deep=True), base_dir=self.base_dir)

    def save(self, path):
        from .schema.io import save_world

        save_world(self.spec, path)
        p = Path(path)
        self.base_dir = p if p.suffix == ".lobs" else p.parent

    @classmethod
    def load(cls, path) -> World:
        from .schema.io import load_world as _load

        p = Path(path)
        return cls(spec=_load(path), base_dir=p if p.is_dir() else p.parent)


def load_world(path) -> World:
    return World.load(path)
