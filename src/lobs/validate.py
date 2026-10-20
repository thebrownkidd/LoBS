"""World validation: structure, measures, and backend capabilities. Reports every problem at once."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends.capabilities import ACTUATOR_CAP, JOINT_CAP, SHAPE_CAP, Capability
from .errors import BackendUnavailable, LobsError, MeasureError


@dataclass
class Issue:
    level: str  # 'error' | 'warning'
    path: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)

    def add(self, level, path, message):
        self.issues.append(Issue(level, path, message))

    @property
    def errors(self):
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self):
        return [i for i in self.issues if i.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self) -> str:
        return (
            "\n".join(f"  [{i.level}] {i.path}: {i.message}" for i in self.issues)
            or "  (no issues)"
        )


_MESH_TYPES = ("convex_mesh", "convex_set", "triangle_mesh")


def required_capabilities(reg, evaluator=None) -> set[Capability]:
    spec = reg.spec
    req = {Capability.RIGID_BODIES}
    for a in spec.actors:
        for b in a.bodies:
            if b.collision_group:
                req.add(Capability.COLLISION_GROUPS)
            for g in b.geoms:
                if g.role == "visual" and g.shape.type in _MESH_TYPES:
                    continue  # visual-only meshes carry no physics
                req.add(SHAPE_CAP[g.shape.type])
                if g.material and g.material.restitution:
                    req.add(Capability.RESTITUTION)
        for j in a.joints:
            req.add(JOINT_CAP[j.type])
            if j.range is not None:
                req.add(Capability.JOINT_LIMITS)
            if j.damping:
                req.add(Capability.JOINT_DAMPING)
            if j.armature:
                req.add(Capability.JOINT_ARMATURE)
        for act in a.actuators:
            req.add(ACTUATOR_CAP[act.kind])
    if spec.contact.disabled_group_pairs:
        req.add(Capability.COLLISION_GROUPS)
    if (spec.contact.default_material.restitution or 0) > 0:
        req.add(Capability.RESTITUTION)
    if evaluator is not None:
        req |= set(evaluator.caps)
        if "accel" in evaluator.needs:
            req.add(Capability.BODY_ACCELERATION)
    return req


def validate_world(world, backend=None) -> ValidationReport:
    from .forces import HIDDEN, compile_laws, hidden_measures
    from .measure import build_evaluator
    from .registry import build_registry

    rep = ValidationReport()
    spec = world.spec
    if spec.timing.dt <= 0 or spec.timing.substeps < 1:
        rep.add("error", "timing", "dt must be > 0 and substeps >= 1")
    try:
        reg = build_registry(spec, getattr(world, "base_dir", None))
    except LobsError as e:
        rep.add("error", "structure", str(e))
        return rep
    for a in reg.spec.actors:
        for b in a.bodies:
            for g in b.geoms:
                if (
                    g.shape.type == "triangle_mesh"
                    and g.role != "visual"
                    and not (a.root == "fixed" and b.parent is None)
                ):
                    rep.add(
                        "error",
                        f"{a.name}/{b.name}",
                        "triangle meshes are allowed on fixed bodies only",
                    )
    for b in reg.bodies:
        if not b.static and b.mass > 0:
            ev = np.linalg.eigvalsh(0.5 * (b.inertia + b.inertia.T))
            if ev[0] <= 0 or ev[0] + ev[1] < ev[2] * (1 - 1e-9):
                rep.add("error", b.path, "inertia tensor is not physically valid (principal moments must be positive and satisfy A + B >= C)")  # fmt: skip
        if not b.static and b.mass <= 0:
            rep.add(
                "error",
                b.path,
                "dynamic body has no mass: give it a collision shape and mass/density",
            )
    for j in reg.joints:
        if j.type == "ball" and j.range is not None:
            rep.add("error", j.path, "joint limits on ball joints are not supported")
    evaluator = None
    try:
        evaluator = build_evaluator(reg, {**spec.measures, **hidden_measures(spec)})
        vec_names = {f"{f.name}.vec" for f in spec.forces if f.vector is not None}
        when_names = {f"{f.name}.when" for f in spec.forces if f.when is not None}
        for n, meta in evaluator.meta().items():
            if not n.startswith(HIDDEN):
                continue
            fname = n[len(HIDDEN) :]
            if n[len(HIDDEN) :] in vec_names:
                fname = n[len(HIDDEN) : -4]
                if tuple(meta["shape"]) != (3,) or meta["kind"] != "real":
                    rep.add(
                        "error",
                        "forces",
                        f"force '{fname}': vector must be a shape-(3,) number measure",
                    )
            elif n[len(HIDDEN) :] in when_names:
                fname = n[len(HIDDEN) : -5]
                if meta["shape"] or meta["kind"] != "bool":
                    rep.add(
                        "error", "forces", f"force '{fname}': when must be a scalar bool measure"
                    )
            elif meta["shape"] or meta["kind"] != "real":
                rep.add("error", "forces", f"force '{n[len(HIDDEN) :]}': magnitude must be a scalar number measure")  # fmt: skip
        compile_laws(reg, spec)
        for f in spec.forces:
            if f.adversary and (f.budget is None or f.budget.peak is None):
                rep.add("warning", "forces", f"adversary '{f.name}' has no budget.peak")
    except (MeasureError, LobsError) as e:
        for part in str(e).split("; "):
            rep.add("error", "measures", part)
    if backend is not None:
        caps = None
        if isinstance(backend, str):
            from . import backends

            try:
                caps = backends.get(backend).capabilities()
            except BackendUnavailable as e:
                rep.add("error", "backend", str(e))
        elif hasattr(backend, "capabilities"):
            caps = backend.capabilities()
        if caps is not None:
            for cap in sorted(required_capabilities(reg, evaluator), key=lambda c: c.value):
                sup = caps.get(cap)
                if sup is None or not sup.ok:
                    rep.add("error", "capability", f"backend does not support {cap.value}")
                elif sup.level == "approximate":
                    rep.add("warning", "capability", f"{cap.value} is approximate: {sup.note}")
    return rep
