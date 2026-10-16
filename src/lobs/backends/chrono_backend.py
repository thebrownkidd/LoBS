"""Project Chrono (PyChrono 10) backend.

Design (plan 9.4). Facts verified against the installed build (see tests/backends/test_conventions.py):
  * ChCollisionShapeBox takes FULL lengths; cylinder axis is Z; the native capsule is unreliable when
    upright, so cylinders and capsules are convex hulls (flagged approximate).
  * Body-level SetPosDt/SetAngVelParent set the COM-frame twist; reference-frame twists are derived
    from the COM frame (v_ref = v_com - w x R c).
  * Contact callback arguments are only valid inside the callback: copy them there. ``R @ f`` is the
    world force on Chrono's body B and column 0 of R is the normal pointing from A to B.
  * Joint state is computed from body states (not from Chrono links) so its semantics equal MuJoCo's.
  * Reset rebuilds the whole ChSystem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import math3d as m3
from ..errors import CapabilityError, LobsError
from ..kinematics import fk_actor
from ..registry import Registry, ResolvedInitial
from ..schema import WorldSpec
from ..state import ContactSet, StateFrame
from .base import PhysicsBackend
from .capabilities import APPROXIMATE, NATIVE, UNSUPPORTED, Capability

try:
    import pychrono as chrono
except ImportError as e:  # pragma: no cover - exercised via lobs.backends.get
    raise ImportError("pychrono is not installed") from e

_PLANE_HALF_EXTENT = 200.0  # plane = 400 x 400 x 1 m box with its top face on the plane
_HULL_SIDES = 24


def _v(x) -> chrono.ChVector3d:
    return chrono.ChVector3d(float(x[0]), float(x[1]), float(x[2]))


def _q(q) -> chrono.ChQuaterniond:
    return chrono.ChQuaterniond(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


def _np(v) -> np.ndarray:
    return np.array([v.x, v.y, v.z])


def _cylinder_points(r: float, h: float, n: int = _HULL_SIDES):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = [(r * np.cos(a), r * np.sin(a)) for a in ang]
    return [(x, y, z) for z in (-h / 2, h / 2) for x, y in ring]


def _capsule_points(r: float, h: float, n: int = _HULL_SIDES, rings: int = 5):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = []
    for sgn in (-1, 1):
        for k in range(rings):
            phi = (np.pi / 2) * k / (rings - 1)
            rr, zz = r * np.cos(phi), r * np.sin(phi)
            if k == rings - 1:
                pts.append((0.0, 0.0, sgn * (h / 2 + zz)))
            else:
                pts += [(rr * np.cos(a), rr * np.sin(a), sgn * (h / 2 + zz)) for a in ang]
    return pts


class ChronoBackend(PhysicsBackend):
    name = "chrono"

    @classmethod
    def available(cls) -> bool:
        return True

    @classmethod
    def engine_version(cls) -> str:
        # Chrono 10 exposes no runtime version attribute; read the conda package metadata.
        meta = sorted((Path(sys.prefix) / "conda-meta").glob("pychrono-*.json"))
        if meta:
            return json.loads(meta[0].read_text(encoding="utf-8")).get("version", meta[0].stem)
        return "unknown"

    @classmethod
    def capabilities(cls):
        caps = {c: NATIVE for c in Capability}
        caps[Capability.SHAPE_PLANE] = APPROXIMATE(
            "plane is a 400x400x1 m box whose top face is the plane"
        )
        caps[Capability.SHAPE_CYLINDER] = APPROXIMATE(
            f"cylinder is a {_HULL_SIDES}-sided convex prism"
        )
        caps[Capability.SHAPE_CAPSULE] = APPROXIMATE(
            "capsule is a convex hull (native capsule contacts are unreliable)"
        )
        caps[Capability.FRICTION] = APPROXIMATE(
            "touching shapes combine friction with Chrono's default rule (the minimum, not the maximum used by"
            " MuJoCo); equal coefficients are exact. NSC contact is impulsive: a body that starts sliding while"
            " already touching can receive a spurious normal impulse"
        )
        caps[Capability.RESTITUTION] = APPROXIMATE("NSC restitution depends on solver iterations")
        caps[Capability.BODY_ACCELERATION] = APPROXIMATE(
            "accelerations are backward differences of velocity"
        )
        for c in (
            Capability.SHAPE_CONVEX_MESH,
            Capability.SHAPE_CONVEX_SET,
            Capability.SHAPE_TRIANGLE_MESH_STATIC,
            Capability.SHAPE_HEIGHTFIELD,
            Capability.JOINT_BALL,
            Capability.JOINT_ARMATURE,
            Capability.JOINT_ACCELERATION,
            Capability.NATIVE_MJCF,
        ):
            caps[c] = UNSUPPORTED
        return caps

    # ------------------------------------------------------------------ construction
    def __init__(self, spec: WorldSpec, registry: Registry, options=None):
        from ..schema.world import ChronoOptions

        self.spec, self.reg = spec, registry
        self.opts = options if options is not None else ChronoOptions()
        self.dt = spec.timing.dt
        self._approx: list[str] = []
        self.system = None
        self._targets = np.zeros(len(registry.actuators))
        self._wrenches: list = []
        self._effort_applied = np.zeros(len(registry.dof_joints))
        self._had_load = False
        self._hinge_q: dict[int, float] = {}
        self._prev_vel: dict[int, tuple] = {}
        self._reject_unsupported()
        self._pair_filter()

    def _reject_unsupported(self):
        for j in self.reg.joints:
            if j.type not in ("hinge", "slide", "fixed"):
                raise CapabilityError(f"chrono backend: joint type '{j.type}' is not supported yet")
            if j.armature:
                raise CapabilityError(
                    f"chrono backend: joint armature is not supported ('{j.path}')"
                )

    def _pair_filter(self):
        """Pairs that must not collide: parent/child bodies and disabled collision-group pairs."""
        reg = self.reg
        self._excluded = {(b.parent, i) for i, b in enumerate(reg.bodies) if b.parent >= 0}
        self._disabled_groups = {tuple(sorted(p)) for p in self.spec.contact.disabled_group_pairs}
        self._groups = [b.collision_group for b in reg.bodies]

    # ------------------------------------------------------------------ build (on reset)
    def _build(self, initial: ResolvedInitial):
        reg, o = self.reg, self.opts
        chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(o.envelope)
        chrono.ChCollisionModel.SetDefaultSuggestedMargin(o.margin)
        s = chrono.ChSystemNSC()
        s.SetGravitationalAcceleration(_v(reg.gravity))
        s.SetCollisionSystemType(getattr(chrono.ChCollisionSystem, f"Type_{o.collision}"))
        s.SetSolverType(getattr(chrono.ChSolver, f"Type_{o.solver}"))
        it = s.GetSolver().AsIterative()
        if it is not None:
            it.SetMaxIterations(o.max_iterations)
        s.SetNumThreads(o.threads, o.threads, 1)

        # world states of every body at the initial configuration (LoBS forward kinematics)
        states = {}
        for ai in range(len(self.spec.actors)):
            free = reg.actors[ai].root_free
            pos, quat = initial.actor_pose[ai]
            v = initial.linvel[ai] if free else np.zeros(3)
            w = initial.angvel[ai] if free else np.zeros(3)
            states.update(fk_actor(reg, ai, (pos, quat), initial.q, initial.qd, v, w))

        self._bodies, self._acc = [], {}
        default = self.spec.contact.default_material
        for gi, info in enumerate(reg.bodies):
            p, q, v, w = states[gi]
            b = chrono.ChBodyAuxRef()
            b.SetTag(gi)
            b.SetFrameRefToAbs(chrono.ChFramed(_v(p), _q(q)))
            b.SetFrameCOMToRef(chrono.ChFramed(_v(info.com), chrono.ChQuaterniond(1, 0, 0, 0)))
            if info.static:
                b.SetFixed(True)
            else:
                b.SetMass(info.mass)
                i3 = info.inertia
                b.SetInertia(chrono.ChMatrix33d(_v(i3[:, 0]), _v(i3[:, 1]), _v(i3[:, 2])))
                rc = m3.quat_to_mat(q) @ info.com
                b.SetPosDt(_v(v + np.cross(w, rc)))
                b.SetAngVelParent(_v(w))
            self._add_shapes(
                b,
                self.spec.actors[info.actor].bodies[reg.actors[info.actor].bodies.index(gi)],
                default,
            )
            s.Add(b)
            if not info.static:
                self._acc[gi] = b.AddAccumulator()
            self._bodies.append(b)

        self._links = {}
        for ji, j in enumerate(reg.joints):
            self._links[ji] = self._add_link(s, ji, j, states, initial)
        if self._excluded_needed():
            self._cb = _Filter(self._excluded, self._disabled_groups, self._groups)
            s.GetCollisionSystem().RegisterBroadphaseCallback(self._cb)
        self.system = s
        self._hinge_q = {
            ji: float(initial.q[reg.joints[ji].dof])
            for ji in reg.dof_joints
            if reg.joints[ji].type == "hinge"
        }
        self._init_q = initial.q.copy()
        self._effort_applied = np.zeros(len(reg.dof_joints))

    def _excluded_needed(self) -> bool:
        return bool(self._excluded) or bool(self._disabled_groups)

    def _material(self, g, default):
        m = chrono.ChContactMaterialNSC()
        mat = g.material
        mu = mat.friction if mat and mat.friction is not None else default.friction
        e = mat.restitution if mat and mat.restitution is not None else default.restitution
        m.SetFriction(0.8 if mu is None else float(mu))
        m.SetRestitution(0.0 if e is None else float(e))
        if e:
            self._approx.append(f"restitution {e} uses the NSC restitution model")
        return m

    def _add_shapes(self, body, bspec, default):
        has_collision = False
        for g in bspec.geoms:
            if g.role == "visual":
                continue
            m = self._material(g, default)
            kind, pos, quat = g.shape.type, np.array(g.pose.pos), np.array(g.pose.quat)
            sh = g.shape
            if kind == "box":
                shape = chrono.ChCollisionShapeBox(m, *[float(x) for x in sh.size])
            elif kind == "sphere":
                shape = chrono.ChCollisionShapeSphere(m, float(sh.radius))
            elif kind in ("cylinder", "capsule"):
                pts = _cylinder_points if kind == "cylinder" else _capsule_points
                vec = chrono.vector_ChVector3d()
                for pt in pts(sh.radius, sh.length):
                    vec.push_back(_v(pt))
                shape = chrono.ChCollisionShapeConvexHull(m, vec)
                self._approx.append(f"{kind} modelled as a convex hull")
            elif kind == "plane":
                shape = chrono.ChCollisionShapeBox(
                    m, 2 * _PLANE_HALF_EXTENT, 2 * _PLANE_HALF_EXTENT, 1.0
                )
                pos, quat = m3.pose_compose(pos, quat, np.array([0.0, 0.0, -0.5]), m3.IDENTITY_QUAT)
                self._approx.append("plane modelled as a large box")
            else:
                raise CapabilityError(f"chrono backend: shape '{kind}' is not supported yet")
            body.AddCollisionShape(shape, chrono.ChFramed(_v(pos), _q(quat)))
            has_collision = True
        body.EnableCollision(has_collision)

    def _add_link(self, s, ji, j, states, initial):
        pc, qc, _, _ = states[j.child]
        rc = m3.quat_to_mat(qc)
        parent, child = self._bodies[j.parent], self._bodies[j.child]
        if j.type == "fixed":
            link = chrono.ChLinkLockLock()
            link.Initialize(parent, child, chrono.ChFramed(_v(pc), _q(qc)))
            s.Add(link)
            return link
        jp = pc + rc @ j.pos
        axis_w = rc @ j.axis
        frame = chrono.ChFramed(_v(jp), _q(m3.mat_to_quat(m3.basis_from_z(axis_w))))
        q0 = float(initial.q[j.dof])
        if j.type == "hinge":
            link = chrono.ChLinkLockRevolute()
            link.Initialize(parent, child, frame)
            if j.range is not None:
                lim = link.LimitRz()
                lim.SetActive(True)
                lim.SetMin(-(j.range[1] - q0))
                lim.SetMax(-(j.range[0] - q0))
        else:
            link = chrono.ChLinkLockPrismatic()
            link.Initialize(parent, child, frame)
            if j.range is not None:
                lim = link.LimitZ()
                lim.SetActive(True)
                lim.SetMin(-(j.range[1] - q0))
                lim.SetMax(-(j.range[0] - q0))
        s.Add(link)
        return link

    # ------------------------------------------------------------------ state reading
    def _body_state(self, gi: int):
        """(p, q, R, v_ref, w) of a body's reference frame, world frame."""
        info = self.reg.bodies[gi]
        if info.static:
            return (
                info.world_pos,
                info.world_quat,
                m3.quat_to_mat(info.world_quat),
                np.zeros(3),
                np.zeros(3),
            )
        b = self._bodies[gi]
        fr = b.GetFrameRefToAbs()
        rot = fr.GetRot()
        q = np.array([rot.e0, rot.e1, rot.e2, rot.e3])
        r = m3.quat_to_mat(q)
        com = b.GetFrameCOMToAbs()
        w = _np(com.GetAngVelParent())
        v = _np(com.GetPosDt()) - np.cross(w, r @ info.com)
        return _np(fr.GetPos()), q, r, v, w

    def _joint_state(self, ji: int, st: dict):
        """(q, qd, axis_w, joint_point_w) from body states; hinge angle is unwrapped."""
        j = self.reg.joints[ji]
        pp, pq, pr, pv, pw = st[j.parent]
        cp, cq, cr, cv, cw = st[j.child]
        cb = self.reg.bodies[j.child]
        c0q = m3.quat_mul(pq, cb.rel_quat)
        c0p = pp + pr @ cb.rel_pos
        axis_w = cr @ j.axis
        jp = cp + cr @ j.pos
        if j.type == "hinge":
            loc = m3.quat_mul(m3.quat_conj(c0q), cq)
            raw = float(m3.wrap_angle(2.0 * np.arctan2(loc[1:] @ j.axis, loc[0])))
            prev = self._hinge_q[ji]
            q = prev + float(m3.wrap_angle(raw - float(m3.wrap_angle(prev))))
            self._hinge_q[ji] = q
            qd = float((cw - pw) @ axis_w)
        else:
            c0v = pv + np.cross(pw, c0p - pp)
            q = float((cp - c0p) @ axis_w)
            qd = float(axis_w @ (cv - c0v))
        return q, qd, axis_w, jp

    def _states(self) -> dict:
        return {gi: self._body_state(gi) for gi in range(len(self.reg.bodies))}

    # ------------------------------------------------------------------ run
    def reset(self, initial: ResolvedInitial, seed):
        self._targets[:] = 0.0
        self._wrenches.clear()
        self._had_load = False
        self._build(initial)
        self._effort_applied = self._compute_efforts(self._states(), apply=False)

    def set_actuator_targets(self, u):
        self._targets = np.asarray(u, dtype=float).copy()

    def add_wrench(self, body, force_w, torque_w, point_w):
        self._wrenches.append(
            (
                int(body),
                np.asarray(force_w, float),
                np.asarray(torque_w, float),
                np.asarray(point_w, float),
            )
        )

    def _compute_efforts(self, st: dict, apply: bool) -> np.ndarray:
        """Actuator efforts (clip -> law -> limit); optionally add them and joint damping as loads."""
        reg = self.reg
        effort = np.zeros(len(reg.dof_joints))
        need = {a.joint for a in reg.actuators} | {
            ji for ji in reg.dof_joints if reg.joints[ji].damping
        }
        for ji in need:
            j = reg.joints[ji]
            q, qd, axis_w, jp = self._joint_state(ji, st)
            tau_act = 0.0
            for ai, a in enumerate(reg.actuators):
                if a.joint != ji:
                    continue
                u = self._targets[ai]
                if a.ctrl_range is not None:
                    u = min(max(u, a.ctrl_range[0]), a.ctrl_range[1])
                if a.kind == "effort":
                    tau = u
                elif a.kind == "position":
                    tau = a.kp * (u - q) - a.kd * qd
                else:
                    tau = a.kd * (u - qd)
                if a.effort_limit is not None:
                    tau = min(max(tau, -a.effort_limit), a.effort_limit)
                tau_act += tau
            effort[j.dof] = tau_act
            tau_total = tau_act - j.damping * qd
            if apply and tau_total != 0.0:
                self._apply_joint_load(j, tau_total, axis_w, jp)
        return effort

    def _apply_joint_load(self, j, tau: float, axis_w, jp):
        for gi, sign in ((j.child, 1.0), (j.parent, -1.0)):
            if self.reg.bodies[gi].static:
                continue
            b, idx = self._bodies[gi], self._acc[gi]
            if j.type == "hinge":
                b.AccumulateTorque(idx, _v(sign * tau * axis_w), False)
            else:
                b.AccumulateForce(idx, _v(sign * tau * axis_w), _v(jp), False)

    def step(self, substeps: int):
        s = self.system
        has_joint_loads = bool(self.reg.actuators) or any(
            self.reg.joints[ji].damping for ji in self.reg.dof_joints
        )
        for _ in range(substeps):
            for gi, idx in self._acc.items():
                self._bodies[gi].EmptyAccumulator(idx)
            if has_joint_loads:
                self._effort_applied = self._compute_efforts(self._states(), apply=True)
            for gi, f, tau, pt in self._wrenches:
                b, idx = self._bodies[gi], self._acc[gi]
                b.AccumulateForce(idx, _v(f), _v(pt), False)
                b.AccumulateTorque(idx, _v(tau), False)
            # Chrono samples accumulated loads at the END of the previous step, so a change made
            # between steps only takes effect after Update() (verified by probe).
            if self._wrenches or has_joint_loads or self._had_load:
                s.Update()
            self._had_load = bool(self._wrenches) or has_joint_loads
            s.DoStepDynamics(self.dt)
        self._wrenches.clear()

    def read_state(self, *, contacts: bool, accelerations: bool) -> StateFrame:
        reg = self.reg
        st = self._states()
        order = reg.dynamic_order
        nd = len(order)
        pos, quat = np.zeros((nd, 3)), np.zeros((nd, 4))
        lin, ang = np.zeros((nd, 3)), np.zeros((nd, 3))
        for k, gi in enumerate(order):
            pos[k], quat[k], _, lin[k], ang[k] = st[gi]
        q = np.zeros(len(reg.dof_joints))
        qd = np.zeros(len(reg.dof_joints))
        for ji in reg.dof_joints:
            j = reg.joints[ji]
            q[j.dof], qd[j.dof], _, _ = self._joint_state(ji, st)
        linacc = angacc = None
        if accelerations:
            linacc, angacc = np.zeros((nd, 3)), np.zeros((nd, 3))
            for k, gi in enumerate(order):
                com = self._bodies[gi].GetFrameCOMToAbs()
                al = _np(com.GetAngAccParent())
                rc = st[gi][2] @ reg.bodies[gi].com
                w = ang[k]
                angacc[k] = al
                linacc[k] = _np(com.GetPosDt2()) - np.cross(al, rc) - np.cross(w, np.cross(w, rc))
        return StateFrame(
            t=float(self.system.GetChTime()), body_pos=pos, body_quat=quat, body_linvel=lin,
            body_angvel=ang, joint_q=q, joint_qd=qd, joint_effort=self._effort_applied.copy(),
            body_linacc=linacc, body_angacc=angacc, joint_qdd=None,
            contacts=self._contacts() if contacts else None,
        )  # fmt: skip

    def _contacts(self) -> ContactSet:
        cb = _Report()
        self.system.GetContactContainer().ReportAllContacts(cb)
        if not cb.rows:
            return ContactSet()
        rows = cb.rows
        return ContactSet(
            body_a=np.array([r[0] for r in rows]), body_b=np.array([r[1] for r in rows]),
            point=np.array([r[2] for r in rows]), normal=np.array([r[3] for r in rows]),
            depth=np.array([r[4] for r in rows]), force_on_b=np.array([r[5] for r in rows]),
            normal_force=np.array([r[6] for r in rows]),
        )  # fmt: skip

    def info(self) -> dict:
        return {
            "engine": "chrono",
            "version": self.engine_version(),
            "options": self.opts.model_dump() if hasattr(self.opts, "model_dump") else {},
            "timestep": self.dt,
            "approximations": sorted(set(self._approx)),
        }

    @property
    def native(self):
        return self.system


class _Filter(chrono.BroadphaseCallback):
    """Veto broadphase pairs: parent/child bodies and disabled collision-group pairs."""

    def __init__(self, excluded, disabled_groups, groups):
        super().__init__()
        self.excluded, self.disabled, self.groups = excluded, disabled_groups, groups

    def OnBroadphase(self, model_a, model_b):
        a = model_a.GetContactable().GetPhysicsItem().GetTag()
        b = model_b.GetContactable().GetPhysicsItem().GetTag()
        if (a, b) in self.excluded or (b, a) in self.excluded:
            return False
        if self.disabled and tuple(sorted((self.groups[a], self.groups[b]))) in self.disabled:
            return False
        return True


class _Report(chrono.ReportContactCallback):
    """Collects contacts; everything is copied here because arguments die after the call."""

    def __init__(self):
        super().__init__()
        self.rows: list = []

    def OnReportContact(self, pa, pb, plane, dist, eff_r, force, torque, obj_a, obj_b, offset):
        if dist > 0:
            return True
        ta = obj_a.GetPhysicsItem().GetTag()
        tb = obj_b.GetPhysicsItem().GetTag()
        cols = np.array([[plane.GetAxisX().x, plane.GetAxisY().x, plane.GetAxisZ().x],
                         [plane.GetAxisX().y, plane.GetAxisY().y, plane.GetAxisZ().y],
                         [plane.GetAxisX().z, plane.GetAxisY().z, plane.GetAxisZ().z]])  # fmt: skip
        f_local = np.array([force.x, force.y, force.z])
        normal, f_world = cols[:, 0], cols @ f_local
        point = np.array([pa.x, pa.y, pa.z])
        if ta <= tb:
            self.rows.append((ta, tb, point, normal, -dist, f_world, f_local[0]))
        else:
            self.rows.append((tb, ta, point, -normal, -dist, -f_world, f_local[0]))
        return True


__all__ = ["ChronoBackend", "LobsError"]
