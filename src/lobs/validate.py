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


def _check_behaviors(rep, reg, spec, meta, BH) -> None:
    from .behavior import registered
    from .schema import behavior as S

    names = [b.name for b in spec.behaviors]
    forces = {f.name for f in spec.forces}
    owners: dict[str, str] = {}
    for b in spec.behaviors:
        loc = f"behavior '{b.name}'"

        def err(msg, loc=loc):
            rep.add("error", "behaviors", f"{loc}: {msg}")

        try:
            if b.on is not None and reg.resolve(b.on).kind != "actor":
                err(f"'on' must be an actor, not '{b.on}'")
        except LobsError as e:
            err(str(e))
        for a in b.owned_actuators():
            try:
                path = reg.actuators[reg.resolve_actuator(a)].path
            except LobsError as e:
                err(str(e))
                continue
            if owners.setdefault(path, b.name) != b.name:
                err(f"actuator '{path}' is also owned by '{owners[path]}'")
        if b.script is not None:
            if not registered(b.script.fn):
                err(f"script '{b.script.fn}' is not registered (use @lobs.behavior)")
            continue
        m = b.machine
        states = [s.name for s in m.states]
        if not states:
            err("the state machine has no states")
            continue
        if len(set(states)) != len(states):
            err("duplicate state names")
        if m.initial is not None and m.initial not in states:
            err(f"initial state '{m.initial}' does not exist")
        reach = {m.initial or states[0]}
        for tr in m.transitions:
            for s in (tr.frm, tr.to):
                if s not in states:
                    err(f"transition {tr.frm} -> {tr.to} refers to unknown state '{s}'")
        grew = True
        while grew:
            grew = False
            for tr in m.transitions:
                if tr.frm in reach and tr.to not in reach:
                    reach.add(tr.to)
                    grew = True
        for s in states:
            if s not in reach:
                rep.add("warning", "behaviors", f"{loc}: state '{s}' is unreachable")
        for key, a in b.actions():
            if isinstance(a, S.Steer):
                try:
                    me = reg.resolve(a.on or b.on or "")
                    if me.kind not in ("actor", "body") or not reg.dyn_bodies_of(me):
                        err(f"steer needs an actor or body ('{key}')")
                    elif all(
                        reg.bodies[i].static or reg.bodies[i].mass <= 0
                        for i in reg.dyn_bodies_of(me)
                    ):
                        err(f"steer: '{me.path}' is static or massless ('{key}')")
                    if a.mode != "hold":
                        if a.target is None:
                            err(f"steer mode '{a.mode}' needs a target ('{key}')")
                        else:
                            reg.resolve(a.target)
                    if a.max_accel <= 0 or a.speed < 0 or a.gain <= 0:
                        err(f"steer needs speed >= 0, max_accel > 0 and gain > 0 ('{key}')")
                except LobsError as e:
                    err(f"{e} ('{key}')")
            elif isinstance(a, S.Force):
                try:
                    reg.resolve(a.on or b.on or "")
                except LobsError as e:
                    err(f"{e} ('{key}')")
            elif isinstance(a, S.EnableForce) and a.force not in forces:
                err(f"enable_force: unknown force '{a.force}' ('{key}')")
            elif isinstance(a, S.EnableBehavior) and a.behavior not in names:
                err(f"enable_behavior: unknown behavior '{a.behavior}' ('{key}')")
            elif isinstance(a, S.SetVar) and a.name not in m.variables:
                err(f"set_var: unknown variable '{a.name}' ('{key}')")
        for key in b.nodes():
            mt = meta.get(f"{BH}{b.name}.{key}")
            if mt is None:
                continue
            want_bool = key.endswith(".when")
            want_vec = key.endswith(".vector")
            if want_bool and (mt["shape"] or mt["kind"] != "bool"):
                err(f"transition condition '{key}' must be a scalar bool measure")
            elif want_vec and (tuple(mt["shape"]) != (3,) or mt["kind"] != "real"):
                err(f"force vector '{key}' must be a shape-(3,) number measure")
            elif key.endswith(".value"):
                a = dict(b.actions())[key[: -len(".value")]]
                var = m.variables.get(a.name) if isinstance(a, S.SetVar) else None
                shape = (3,) if var is not None and not isinstance(var, (int, float)) else ()
                if mt["kind"] != "real" or tuple(mt["shape"]) != shape:
                    err(f"value '{key}' must be a {'vector' if shape else 'scalar'} number measure")


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
        from .behavior import HIDDEN as BH
        from .behavior import hidden_measures as beh_hidden

        evaluator = build_evaluator(
            reg, {**spec.measures, **hidden_measures(spec), **beh_hidden(spec)}
        )
        _check_behaviors(rep, reg, spec, evaluator.meta(), BH)
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
