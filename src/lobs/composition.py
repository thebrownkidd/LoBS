"""World composition: merge one world into another (`World.include`)."""

from __future__ import annotations

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

from . import math3d as m3
from .errors import LobsError
from .schema import Pose
from .schema.measure import ConstNode, CustomNode, OpNode, PropNode, RefNode, RelNode, TimeNode


def _map_path(path: str, prefix: str) -> str:
    base, sep, suffix = path.partition(":")
    tail = (sep + suffix) if sep else ""
    if base == "world":
        return path
    if base.startswith("world/"):
        return f"world/{prefix}.{base[len('world/') :]}{tail}"
    head, slash, rest = base.partition("/")
    return f"{prefix}.{head}{slash}{rest}{tail}"


def _rewrite(node, prefix: str):
    if isinstance(node, PropNode):
        args = dict(node.args)
        if "with" in args and args["with"]:
            args["with"] = _map_path(args["with"], prefix)
        return node.model_copy(
            update={"target": _map_path(node.target, prefix) if node.target else None, "args": args}
        )
    if isinstance(node, RelNode):
        upd = {"things": [_map_path(t, prefix) for t in node.things]}
        if node.frame and node.frame not in ("a", "b", "v", "world"):
            upd["frame"] = _map_path(node.frame, prefix)
        return node.model_copy(update=upd)
    if isinstance(node, RefNode):
        return node.model_copy(update={"name": f"{prefix}.{node.name}"})
    if isinstance(node, OpNode):
        return node.model_copy(update={"inputs": [_rewrite(i, prefix) for i in node.inputs]})
    if isinstance(node, CustomNode):
        return node.model_copy(
            update={"inputs": {k: _rewrite(v, prefix) for k, v in node.inputs.items()}}
        )
    if isinstance(node, (ConstNode, TimeNode)):
        return node
    raise LobsError(f"cannot rewrite measure node {type(node).__name__}")


def include_world(host, other, prefix: str, pose=None) -> None:
    from .world import World

    if not prefix or "/" in prefix or ":" in prefix:
        raise LobsError("prefix must be a non-empty name without '/' or ':'")
    guest = other if isinstance(other, World) else World.load(other)
    spec = guest.spec.model_copy(deep=True)
    hs = host.spec
    existing = {a.name for a in hs.actors} | {a.name for a in hs.anchors}
    if pose is None:
        pose = Pose()
    elif not isinstance(pose, Pose):
        raise LobsError("pose must be a lobs.Pose")
    p0, q0 = np.array(pose.pos), np.array(pose.quat)

    def place(p: Pose) -> Pose:
        pos, quat = m3.pose_compose(p0, q0, np.array(p.pos), np.array(p.quat))
        return Pose(pos=tuple(float(x) for x in pos), quat=tuple(float(x) for x in quat))

    for a in spec.actors:
        new = f"{prefix}.{a.name}"
        if new in existing:
            raise LobsError(f"include: name '{new}' already exists in the host world")
        a.name = new
        a.pose = place(a.pose)
        if a.initial.pose is not None:
            a.initial.pose = place(a.initial.pose)
        hs.actors.append(a)
    for an in spec.anchors:
        new = f"{prefix}.{an.name}"
        if new in existing:
            raise LobsError(f"include: name '{new}' already exists in the host world")
        an.name = new
        an.pose = place(an.pose)
        hs.anchors.append(an)
    for f in spec.forces:
        f.name = f"{prefix}.{f.name}"
        f.on = _map_path(f.on, prefix)
        f.toward = None if f.toward is None else _map_path(f.toward, prefix)
        f.axis = None if f.axis is None else (_map_path(f.axis[0], prefix), f.axis[1])
        f.magnitude = None if f.magnitude is None else _rewrite(f.magnitude, prefix)
        f.vector = None if f.vector is None else _rewrite(f.vector, prefix)
        f.when = None if f.when is None else _rewrite(f.when, prefix)
        if any(x.name == f.name for x in hs.forces):
            raise LobsError(f"include: force '{f.name}' already exists")
        hs.forces.append(f)
    for name, node in spec.measures.items():
        new = f"{prefix}.{name}"
        if new in hs.measures:
            raise LobsError(f"include: measure '{new}' already exists")
        hs.measures[new] = _rewrite(node, prefix)
    if tuple(spec.gravity) != tuple(hs.gravity):
        warnings.warn(
            f"include: guest gravity {spec.gravity} differs; the host's {hs.gravity} is used",
            stacklevel=3,
        )
    hs.imports.extend(spec.imports)
    # assets referenced by the guest must resolve from the host's directory
    refs = {
        g.shape.file
        for a in spec.actors
        for b in a.bodies
        for g in b.geoms
        if hasattr(g.shape, "file")
    }
    refs |= {
        f
        for a in spec.actors
        for b in a.bodies
        for g in b.geoms
        if hasattr(g.shape, "files")
        for f in g.shape.files
    }
    if refs:
        if guest.base_dir is None:
            raise LobsError(
                "include: the guest world references assets but has no directory (save it first)"
            )
        if host.base_dir is None:
            host.base_dir = Path(tempfile.mkdtemp(prefix="lobs_world_"))
        for rel in sorted(refs):
            dst = host.base_dir / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(guest.base_dir) / rel, dst)
