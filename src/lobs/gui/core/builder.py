"""World-building operations for the GUI. Every function takes a spec (so it runs inside ``Session.edit``)
and reuses the library builder; nothing here knows about Qt."""

from __future__ import annotations

import copy

from ...errors import LobsError
from ...world import World


def unique_name(existing, base: str) -> str:
    names = set(existing)
    if base not in names:
        return base
    i = 2
    while f"{base}{i}" in names:
        i += 1
    return f"{base}{i}"


def _actor_names(spec):
    return [a.name for a in spec.actors] + [a.name for a in spec.anchors]


def add_primitive(
    spec, kind: str, pos=(0.0, 0.0, 1.0), name: str | None = None, fixed: bool = False
) -> str:
    """Add a box / sphere / capsule / cylinder actor; returns its name."""
    w = World(spec=spec)
    name = unique_name(_actor_names(spec), name or kind)
    if kind == "box":
        w.add_box(name, (0.5, 0.5, 0.5), mass=1.0, pos=tuple(pos), fixed=fixed)
    elif kind == "sphere":
        w.add_sphere(name, 0.25, mass=1.0, pos=tuple(pos), fixed=fixed)
    elif kind in ("capsule", "cylinder"):
        from ... import schema

        shape = (
            schema.Capsule(radius=0.15, length=0.6)
            if kind == "capsule"
            else schema.Cylinder(radius=0.2, length=0.5)
        )
        a = w.add_actor(name, root="fixed" if fixed else "free", pos=tuple(pos))
        a.add_body("body", geoms=[shape], mass=1.0)
    elif kind == "ground":
        w.add_box(name, (20.0, 20.0, 0.2), pos=(0.0, 0.0, -0.1), fixed=True)
    else:
        raise LobsError(f"unknown primitive '{kind}'")
    return name


def add_world_anchor(spec, pos=(0.0, 0.0, 1.0), name: str | None = None) -> str:
    name = unique_name(_actor_names(spec), name or "anchor")
    World(spec=spec).anchor(name, pos=tuple(pos))
    return name


def add_force(
    spec, on: str, name: str | None = None, vector=(0.0, 0.0, 10.0), adversary: bool = False
) -> str:
    from ...schema.world import ForceBudget

    w = World(spec=spec)
    name = unique_name(
        [f.name for f in spec.forces], name or ("adversary" if adversary else "force")
    )
    budget = ForceBudget(peak=10.0, impulse=5.0) if adversary else None
    w.add_force(name, on=on, vector=list(vector), adversary=adversary, budget=budget)
    return name


def delete_item(spec, path: str) -> None:
    """Delete by tree path: actor, force/<n>, behavior/<n>, world/<anchor>, measure/<n>."""
    head, _, rest = path.partition("/")
    if head == "force":
        spec.forces = [f for f in spec.forces if f.name != rest]
    elif head == "behavior":
        spec.behaviors = [b for b in spec.behaviors if b.name != rest]
    elif head == "measure":
        spec.measures.pop(rest, None)
        spec.metadata.get("gui", {}).get("expr", {}).pop(rest, None)
    elif head == "world":
        spec.anchors = [a for a in spec.anchors if a.name != rest]
    elif not rest:
        spec.actors = [a for a in spec.actors if a.name != path]
        spec.forces = [
            f
            for f in spec.forces
            if f.on.split("/")[0] != path and (f.toward or "").split("/")[0] != path
        ]
        spec.behaviors = [b for b in spec.behaviors if b.on != path]
    else:
        raise LobsError(f"cannot delete '{path}' here (delete the whole actor instead)")


def duplicate_actor(spec, name: str, offset=(0.5, 0.0, 0.0)) -> str:
    src = next(a for a in spec.actors if a.name == name)
    new = copy.deepcopy(src)
    new.name = unique_name(_actor_names(spec), f"{name}_copy")
    new.pose.pos = tuple(p + o for p, o in zip(new.pose.pos, offset, strict=True))
    if new.initial.pose is not None:
        new.initial.pose.pos = tuple(
            p + o for p, o in zip(new.initial.pose.pos, offset, strict=True)
        )
    spec.actors.append(new)
    return new.name


def set_actor_position(spec, name: str, pos) -> None:
    a = next(x for x in spec.actors if x.name == name)
    a.pose.pos = tuple(float(x) for x in pos)
    if a.initial.pose is not None:
        a.initial.pose.pos = a.pose.pos


def use_current_as_initial(spec, sim) -> int:
    """Store the live pose and velocity of every free actor as its initial state. Returns the count."""
    from ...schema import Pose

    n = 0
    frame = sim.state()
    for a in spec.actors:
        info = sim.reg.resolve(a.name)
        actor = sim.reg.actors[info.index]
        root = sim.reg.bodies[actor.bodies[0]]
        if root.static or a.root != "free":
            continue
        k = root.dyn
        a.initial.pose = Pose(
            pos=tuple(float(x) for x in frame.body_pos[k]),
            quat=tuple(float(x) for x in frame.body_quat[k]),
        )
        a.initial.linvel = tuple(float(x) for x in frame.body_linvel[k])
        a.initial.angvel = tuple(float(x) for x in frame.body_angvel[k])
        n += 1
    return n


def resolve_object(spec, path: str):
    """The spec model a tree path refers to: ``world`` / actor / ``actor/part`` / ``world/anchor`` /
    ``force/name`` / ``behavior/name``. Returns ``(object, kind)`` or ``(None, '')``."""
    if path in ("", "world"):
        return spec, "world"
    head, _, rest = path.partition("/")
    if head == "force":
        return next((f for f in spec.forces if f.name == rest), None), "force"
    if head == "behavior":
        return next((b for b in spec.behaviors if b.name == rest), None), "behavior"
    if head == "measure":
        return None, "measure"
    if head == "world":
        return next((a for a in spec.anchors if a.name == rest), None), "anchor"
    actor = next((a for a in spec.actors if a.name == head), None)
    if actor is None:
        return None, ""
    if not rest:
        return actor, "actor"
    for kind, items in (
        ("body", actor.bodies),
        ("joint", actor.joints),
        ("anchor", actor.anchors),
        ("actuator", actor.actuators),
    ):
        hit = next((i for i in items if i.name == rest), None)
        if hit is not None:
            return hit, kind
    return None, ""
