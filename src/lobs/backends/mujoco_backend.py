"""MuJoCo backend. Model is built with ``mujoco.MjSpec`` (no XML strings).

Conventions (plan section 4 / 9.3):
  * MuJoCo body frame == LoBS reference frame; mass props are passed explicitly (never inferred).
  * Twists use ``mj_objectVelocity(mjOBJ_XBODY)`` (reference frame, not the inertial frame).
  * Free-joint qvel: linear is world frame; angular is body-local (converted on reset).
  * State is re-derived (kinematics / forward) before reading, because ``mj_step`` leaves xpos stale.
"""

from __future__ import annotations

import numpy as np

from .. import math3d as m3
from ..errors import CapabilityError, LobsError
from ..inertia import principal_axes
from ..registry import Registry, ResolvedInitial
from ..schema import WorldSpec
from ..state import ContactSet, StateFrame
from .base import PhysicsBackend
from .capabilities import APPROXIMATE, NATIVE, UNSUPPORTED, Capability

try:
    import mujoco
except ImportError as e:  # pragma: no cover - exercised via lobs.backends.get
    raise ImportError("mujoco is not installed") from e


def _zeta(e: float) -> float:
    """Damping ratio giving restitution ~e for a spring-damper contact."""
    if e <= 0:
        return 1.0
    if e >= 1:
        return 0.01
    ln = np.log(e)
    return max(0.01, float(-ln / np.sqrt(np.pi**2 + ln**2)))


_INTEGRATORS = {
    "euler": "mjINT_EULER",
    "rk4": "mjINT_RK4",
    "implicit": "mjINT_IMPLICIT",
    "implicitfast": "mjINT_IMPLICITFAST",
}
_CONES = {"pyramidal": "mjCONE_PYRAMIDAL", "elliptic": "mjCONE_ELLIPTIC"}


class MujocoBackend(PhysicsBackend):
    name = "mujoco"

    @classmethod
    def available(cls) -> bool:
        return True

    @classmethod
    def engine_version(cls) -> str:
        return mujoco.__version__

    @classmethod
    def capabilities(cls):
        caps = {c: NATIVE for c in Capability}
        caps[Capability.RESTITUTION] = APPROXIMATE(
            "restitution is emulated through contact solref damping ratio and mixed with the partner geom"
        )
        for c in (
            Capability.SHAPE_CONVEX_MESH,
            Capability.SHAPE_CONVEX_SET,
            Capability.SHAPE_TRIANGLE_MESH_STATIC,
            Capability.SHAPE_HEIGHTFIELD,
            Capability.JOINT_BALL,
            Capability.NATIVE_MJCF,
        ):
            caps[c] = UNSUPPORTED
        return caps

    # ------------------------------------------------------------------ build
    def __init__(self, spec: WorldSpec, registry: Registry, options=None):
        from ..schema.world import MujocoOptions

        self.reg = registry
        self.opts = options if options is not None else MujocoOptions()
        self._approx: list[str] = []
        self._targets = np.zeros(len(registry.actuators))
        self._wrenches: list = []
        self._effort_applied = np.zeros(len(registry.actuators))
        ms = self._build_spec(spec)
        self.model = ms.compile()
        self.data = mujoco.MjData(self.model)
        self._index()
        self._mjspec = ms

    def to_xml(self) -> str:
        """Debug helper: the compiled model as MJCF."""
        return self._mjspec.to_xml()

    def _build_spec(self, spec: WorldSpec):
        reg, o = self.reg, self.opts
        ms = mujoco.MjSpec()
        ms.compiler.degree = False  # MjSpec defaults to degrees for joint ranges; LoBS is radians
        ms.option.timestep = spec.timing.dt
        ms.option.gravity = list(reg.gravity)
        ms.option.integrator = getattr(mujoco.mjtIntegrator, _INTEGRATORS[o.integrator])
        ms.option.cone = getattr(mujoco.mjtCone, _CONES[o.cone])
        ms.option.iterations = o.iterations
        ms.option.tolerance = o.tolerance
        default_mat = spec.contact.default_material
        bodies = {}
        for ai, actor in enumerate(spec.actors):
            for gi in reg.actors[ai].bodies:
                b = reg.bodies[gi]
                bspec = actor.bodies[reg.actors[ai].bodies.index(gi)]
                if b.parent < 0:
                    parent = ms.worldbody
                    pos, quat = m3.pose_compose(
                        actor.pose.pos, actor.pose.quat, b.rel_pos, b.rel_quat
                    )
                else:
                    parent = bodies[b.parent]
                    pos, quat = b.rel_pos, b.rel_quat
                mb = parent.add_body(name=b.path, pos=list(pos), quat=list(quat))
                bodies[gi] = mb
                if b.parent < 0 and actor.root == "free":
                    mb.add_freejoint(name=f"{b.path}/__free")
                if b.parent >= 0:
                    j = reg.joints[b.joint]
                    if j.type != "fixed":
                        self._add_joint(mb, j)
                self._set_inertial(mb, b)
                for g in bspec.geoms:
                    self._add_geom(mb, g, default_mat)
        # Collision filtering. Contract: parent/child bodies never collide. MuJoCo's own parent-child
        # filter is skipped when the parent is welded to the world, so exclude explicitly.
        pairs = {(b.parent, i) for i, b in enumerate(reg.bodies) if b.parent >= 0}
        disabled = {tuple(sorted(p)) for p in spec.contact.disabled_group_pairs}
        if disabled:
            n = len(reg.bodies)
            for i in range(n):
                for j in range(i + 1, n):
                    key = tuple(
                        sorted((reg.bodies[i].collision_group, reg.bodies[j].collision_group))
                    )
                    if key in disabled:
                        pairs.add((i, j))
        for i, j in sorted(pairs):
            ex = ms.add_exclude()
            ex.bodyname1, ex.bodyname2 = reg.bodies[i].path, reg.bodies[j].path
        for a in reg.actuators:
            self._add_actuator(ms, a)
        return ms

    def _set_inertial(self, mb, b):
        mb.explicitinertial = True
        mb.mass = max(b.mass, 0.0)
        if b.mass > 0:
            diag, q = principal_axes(b.inertia)
            mb.ipos = list(b.com)
            mb.iquat = list(q)
            mb.inertia = list(diag)

    def _add_joint(self, mb, j):
        kind = mujoco.mjtJoint.mjJNT_HINGE if j.type == "hinge" else mujoco.mjtJoint.mjJNT_SLIDE
        if j.type not in ("hinge", "slide"):
            raise CapabilityError(f"mujoco backend: joint type '{j.type}' is not supported yet")
        mj = mb.add_joint(name=j.path, type=kind, pos=list(j.pos), axis=list(j.axis))
        mj.damping = [j.damping, 0.0, 0.0]  # [linear, quadratic, cubic...] in mujoco >= 3.16
        mj.armature = j.armature
        if j.range is not None:
            mj.range = list(j.range)
            mj.limited = mujoco.mjtLimited.mjLIMITED_TRUE

    def _add_geom(self, mb, g, default_mat):
        shapes = {
            "box": (mujoco.mjtGeom.mjGEOM_BOX, lambda s: [x / 2 for x in s.size]),
            "sphere": (mujoco.mjtGeom.mjGEOM_SPHERE, lambda s: [s.radius, 0, 0]),
            "cylinder": (mujoco.mjtGeom.mjGEOM_CYLINDER, lambda s: [s.radius, s.length / 2, 0]),
            "capsule": (mujoco.mjtGeom.mjGEOM_CAPSULE, lambda s: [s.radius, s.length / 2, 0]),
            "plane": (mujoco.mjtGeom.mjGEOM_PLANE, lambda s: [0, 0, 1]),
        }
        kind = g.shape.type
        if kind not in shapes:
            raise CapabilityError(f"mujoco backend: shape '{kind}' is not supported yet")
        gtype, size = shapes[kind]
        mat = g.material
        mu = mat.friction if mat and mat.friction is not None else default_mat.friction
        e = mat.restitution if mat and mat.restitution is not None else default_mat.restitution
        e = e or 0.0
        if e > 0:
            self._approx.append(f"restitution {e} emulated via solref damping ratio")
        mg = mb.add_geom(
            type=gtype, size=size(g.shape), pos=list(g.pose.pos), quat=list(g.pose.quat)
        )
        mg.friction = [mu if mu is not None else 0.8, 0.005, 0.0001]
        mg.solref = [self.opts.solref_timeconst, _zeta(e)]
        if g.rgba is not None:
            mg.rgba = list(g.rgba)
        if g.role == "visual":
            mg.contype = 0
            mg.conaffinity = 0
            mg.group = 1
        elif g.role == "collision":
            mg.group = 3

    def _add_actuator(self, ms, a):
        ja = self.reg.joints[a.joint]
        act = ms.add_actuator(name=a.path, target=ja.path, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        if a.kind == "effort":
            act.gainprm[0] = 1.0
            act.biasprm[:3] = [0.0, 0.0, 0.0]
        elif a.kind == "position":
            act.gainprm[0] = a.kp
            act.biasprm[:3] = [0.0, -a.kp, -a.kd]
        else:  # velocity
            act.gainprm[0] = a.kd
            act.biasprm[:3] = [0.0, 0.0, -a.kd]
        if a.ctrl_range is not None:
            act.ctrlrange = list(a.ctrl_range)
            act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        if a.effort_limit is not None:
            act.forcerange = [-a.effort_limit, a.effort_limit]
            act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    def _index(self):
        m, reg = self.model, self.reg
        oid = mujoco.mjtObj
        self._mj_body = np.array([mujoco.mj_name2id(m, oid.mjOBJ_BODY, b.path) for b in reg.bodies])
        self._dyn_ids = self._mj_body[reg.dynamic_order] if reg.dynamic_order else np.zeros(0, int)
        self._body_of_mj = {int(mb): i for i, mb in enumerate(self._mj_body)}
        self._geom_body = np.array([self._body_of_mj.get(int(b), -1) for b in m.geom_bodyid])
        self._qpos_adr, self._dof_adr = [], []
        for ji in reg.dof_joints:
            jid = mujoco.mj_name2id(m, oid.mjOBJ_JOINT, reg.joints[ji].path)
            self._qpos_adr.append(int(m.jnt_qposadr[jid]))
            self._dof_adr.append(int(m.jnt_dofadr[jid]))
        self._qpos_adr = np.array(self._qpos_adr, int)
        self._dof_adr = np.array(self._dof_adr, int)
        self._free = {}
        for ai, a in enumerate(reg.actors):
            if a.root_free:
                jid = mujoco.mj_name2id(
                    m, oid.mjOBJ_JOINT, f"{reg.bodies[a.bodies[0]].path}/__free"
                )
                self._free[ai] = (int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]))
        self._act_dof = np.array([reg.joints[a.joint].dof for a in reg.actuators], int)

    # ------------------------------------------------------------------ run
    def reset(self, initial: ResolvedInitial, seed):
        m, d, reg = self.model, self.data, self.reg
        mujoco.mj_resetData(m, d)
        self._targets[:] = 0.0
        self._wrenches.clear()
        spec_actors = reg.spec.actors
        for ai, actor in enumerate(spec_actors):
            pos, quat = initial.actor_pose[ai]
            root = reg.bodies[reg.actors[ai].bodies[0]]
            if ai in self._free:
                qa, da = self._free[ai]
                p, q = m3.pose_compose(pos, quat, root.rel_pos, root.rel_quat)
                d.qpos[qa : qa + 3] = p
                d.qpos[qa + 3 : qa + 7] = q
                d.qvel[da : da + 3] = initial.linvel[ai]
                d.qvel[da + 3 : da + 6] = m3.quat_to_mat(q).T @ initial.angvel[ai]
            else:
                if not (
                    np.allclose(pos, actor.pose.pos)
                    and m3.quat_same_rotation(quat, actor.pose.quat)
                ):
                    raise LobsError(f"cannot move fixed actor '{actor.name}' at reset")
        for k in range(len(reg.dof_joints)):
            d.qpos[self._qpos_adr[k]] = initial.q[k]
            d.qvel[self._dof_adr[k]] = initial.qd[k]
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        self._effort_applied = d.actuator_force.copy()

    def set_actuator_targets(self, u):
        self._targets = np.asarray(u, dtype=float).copy()

    def add_wrench(self, body, force_w, torque_w, point_w):
        self._wrenches.append(
            (
                int(self._mj_body[body]),
                np.asarray(force_w, float),
                np.asarray(torque_w, float),
                np.asarray(point_w, float),
            )
        )

    def step(self, substeps: int):
        m, d = self.model, self.data
        if m.nu:
            d.ctrl[:] = self._targets
        for _ in range(substeps):
            d.qfrc_applied[:] = 0.0
            for body, f, tau, pt in self._wrenches:
                mujoco.mj_applyFT(m, d, f, tau, pt, body, d.qfrc_applied)
            mujoco.mj_step(m, d)
        self._wrenches.clear()
        # effort actually applied during the last substep (a later mj_forward would re-evaluate it
        # at the end-of-step state, making the value depend on what else was read)
        self._effort_applied = d.actuator_force.copy()

    def read_state(self, *, contacts: bool, accelerations: bool) -> StateFrame:
        m, d = self.model, self.data
        if contacts or accelerations:
            mujoco.mj_forward(m, d)
        else:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d)
        ids = self._dyn_ids
        nd = len(ids)
        pos, quat = d.xpos[ids].copy(), d.xquat[ids].copy()
        lin, ang = np.zeros((nd, 3)), np.zeros((nd, 3))
        res = np.zeros(6)
        for k, mid in enumerate(ids):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, int(mid), res, 0)
            ang[k], lin[k] = res[:3], res[3:]
        q = d.qpos[self._qpos_adr].copy()
        qd = d.qvel[self._dof_adr].copy()
        effort = np.zeros(len(q))
        for ai, dof in enumerate(self._act_dof):
            effort[dof] += self._effort_applied[ai]
        linacc = angacc = qdd = None
        if accelerations:
            mujoco.mj_rnePostConstraint(m, d)
            linacc, angacc = np.zeros((nd, 3)), np.zeros((nd, 3))
            for k, mid in enumerate(ids):
                mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_XBODY, int(mid), res, 0)
                angacc[k] = res[:3]
                linacc[k] = res[3:] + self.reg.gravity  # proper -> coordinate acceleration
            qdd = d.qacc[self._dof_adr].copy()
        cs = self._contacts() if contacts else None
        return StateFrame(
            t=float(d.time), body_pos=pos, body_quat=quat, body_linvel=lin, body_angvel=ang,
            joint_q=q, joint_qd=qd, joint_effort=effort, body_linacc=linacc, body_angacc=angacc,
            joint_qdd=qdd, contacts=cs,
        )  # fmt: skip

    def _contacts(self) -> ContactSet:
        m, d = self.model, self.data
        rows = []
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0 or c.exclude:
                continue
            b1, b2 = int(self._geom_body[c.geom1]), int(self._geom_body[c.geom2])
            mujoco.mj_contactForce(m, d, i, f6)
            frame = np.asarray(c.frame).reshape(3, 3)
            normal = frame[0].copy()
            force_on_2 = frame.T @ f6[:3]
            if b1 <= b2:
                rows.append((b1, b2, c.pos.copy(), normal, -c.dist, force_on_2, f6[0]))
            else:
                rows.append((b2, b1, c.pos.copy(), -normal, -c.dist, -force_on_2, f6[0]))
        if not rows:
            return ContactSet()
        return ContactSet(
            body_a=np.array([r[0] for r in rows]), body_b=np.array([r[1] for r in rows]),
            point=np.array([r[2] for r in rows]), normal=np.array([r[3] for r in rows]),
            depth=np.array([r[4] for r in rows]), force_on_b=np.array([r[5] for r in rows]),
            normal_force=np.array([r[6] for r in rows]),
        )  # fmt: skip

    def info(self) -> dict:
        return {
            "engine": "mujoco",
            "version": mujoco.__version__,
            "options": self.opts.model_dump() if hasattr(self.opts, "model_dump") else {},
            "timestep": float(self.model.opt.timestep),
            "approximations": sorted(set(self._approx)),
        }

    @property
    def native(self):
        return self.model, self.data
