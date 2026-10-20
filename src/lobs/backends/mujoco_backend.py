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
from .. import native
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
        caps[Capability.SHAPE_CONVEX_MESH] = APPROXIMATE(
            "mesh collision uses the convex hull of the mesh"
        )
        caps[Capability.SHAPE_CONVEX_SET] = APPROXIMATE(
            "each hull is a separate mesh geom (convex hull collision)"
        )
        for c in (
            Capability.SHAPE_TRIANGLE_MESH_STATIC,
            Capability.SHAPE_HEIGHTFIELD,
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
        ms = self._ms = mujoco.MjSpec()
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
        kinds = {
            "hinge": mujoco.mjtJoint.mjJNT_HINGE,
            "slide": mujoco.mjtJoint.mjJNT_SLIDE,
            "ball": mujoco.mjtJoint.mjJNT_BALL,
        }
        if j.type not in kinds:
            raise CapabilityError(f"mujoco backend: joint type '{j.type}' is not supported yet")
        if j.type == "ball":
            mj = mb.add_joint(name=j.path, type=kinds["ball"], pos=list(j.pos))
        else:
            mj = mb.add_joint(name=j.path, type=kinds[j.type], pos=list(j.pos), axis=list(j.axis))
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
        if kind in ("convex_mesh", "convex_set", "triangle_mesh") and g.role == "visual":
            return  # visual-only meshes carry no physics; skipped (rendering is a viewer concern)
        if kind in ("convex_mesh", "convex_set"):
            self._add_mesh_geoms(mb, g, default_mat)
            return
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

    def _add_mesh_geoms(self, mb, g, default_mat):
        from ..assets import load_mesh

        files = [g.shape.file] if g.shape.type == "convex_mesh" else list(g.shape.files)
        mat = g.material
        mu = mat.friction if mat and mat.friction is not None else default_mat.friction
        e = (
            mat.restitution if mat and mat.restitution is not None else default_mat.restitution
        ) or 0.0
        if e > 0:
            self._approx.append(f"restitution {e} emulated via solref damping ratio")
        for f in files:
            verts, faces = load_mesh(self.reg.base_dir, f)
            self._n_meshes = getattr(self, "_n_meshes", 0) + 1
            name = f"mesh{self._n_meshes}"
            mm = self._ms.add_mesh(name=name)
            mm.uservert = verts.astype(np.float32).ravel()
            mm.userface = faces.astype(np.int32).ravel()
            mg = mb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
                             pos=list(g.pose.pos), quat=list(g.pose.quat))  # fmt: skip
            mg.friction = [mu if mu is not None else 0.8, 0.005, 0.0001]
            mg.solref = [self.opts.solref_timeconst, _zeta(e)]
            if g.role == "collision":
                mg.group = 3
        if len(files) > 1:
            self._approx.append("convex set: each hull is an independent collision geom")

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
        self._jadr = {}  # joint index -> (qpos address, dof address)
        for ji in reg.dof_joints:
            jid = mujoco.mj_name2id(m, oid.mjOBJ_JOINT, reg.joints[ji].path)
            self._jadr[ji] = (int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]))
        self._ball_slots = tuple(
            reg.joints[ji].dof for ji in reg.dof_joints if reg.joints[ji].type == "ball"
        )
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
        for ji in reg.dof_joints:
            j = reg.joints[ji]
            qa, da = self._jadr[ji]
            if j.type == "ball":
                d.qpos[qa : qa + 4] = m3.quat_exp(initial.q[j.dof : j.dof + 3])
                d.qvel[da : da + 3] = initial.qd[j.dof : j.dof + 3]
            else:
                d.qpos[qa] = initial.q[j.dof]
                d.qvel[da] = initial.qd[j.dof]
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)
        self._effort_applied = d.actuator_force.copy()
        if getattr(self, "_kernel", None) is not None:
            self._kernel.set_effort(np.asarray(self._effort_applied, dtype=float))

    def set_actuator_targets(self, u):
        self._targets = np.asarray(u, dtype=float).copy()
        k = getattr(self, "_kernel", None)
        if k is not None:
            k.set_targets(self._targets)

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
        if not getattr(self, "_prepared", False):
            self._prep_read()
        k = self._kernel
        if k is not None:
            if m.nu:
                k.set_targets(self._targets)
            if self._wrenches:
                arr = np.array(
                    [[body, *f, *tau, *pt] for body, f, tau, pt in self._wrenches], dtype=float
                )
                self._wrenches.clear()
            else:
                arr = self._no_wrenches
            k.step(substeps, arr)
            if m.nu:
                self._effort_applied = d.actuator_force.copy()
            return
        if m.nu:
            d.ctrl[:] = self._targets
        if not self._wrenches:
            d.qfrc_applied[:] = 0.0
            mujoco.mj_step(
                m, d, substeps
            )  # one native call; identical to stepping substep by substep
        else:
            wrenches = self._merged_wrenches()
            for _ in range(substeps):
                d.qfrc_applied[:] = 0.0
                for body, f, tau, pt in wrenches:
                    mujoco.mj_applyFT(m, d, f, tau, pt, body, d.qfrc_applied)
                mujoco.mj_step(m, d)
            self._wrenches.clear()
        # effort actually applied during the last substep (a later mj_forward would re-evaluate it
        # at the end-of-step state, making the value depend on what else was read)
        self._effort_applied = d.actuator_force.copy() if m.nu else self._effort_applied

    def _merged_wrenches(self):
        """Wrenches on the same body at the same point are summed (one native call instead of one each)."""
        if len(self._wrenches) < 2:
            return self._wrenches
        merged: dict = {}
        for body, f, tau, pt in self._wrenches:
            key = (body, pt.tobytes())
            hit = merged.get(key)
            if hit is None:
                merged[key] = [body, f.copy(), tau.copy(), pt]
            else:
                hit[1] += f
                hit[2] += tau
        return list(merged.values())

    def _prep_read(self) -> None:
        """Index arrays for the vectorised state read (built once)."""
        reg, d = self.reg, self.data
        ids = np.asarray(self._dyn_ids, dtype=int)
        self._ids = ids
        self._no_wrenches = np.zeros((0, 10))
        self._rootid = np.asarray(self.model.body_rootid)[ids]
        simple = [ji for ji in reg.dof_joints if reg.joints[ji].type != "ball"]
        self._s_dof = np.array([reg.joints[ji].dof for ji in simple], dtype=int)
        self._s_qa = np.array([self._jadr[ji][0] for ji in simple], dtype=int)
        self._s_da = np.array([self._jadr[ji][1] for ji in simple], dtype=int)
        self._balls = [ji for ji in reg.dof_joints if reg.joints[ji].type == "ball"]
        self._act_dof_arr = np.asarray(self._act_dof, dtype=int)
        self._act_unique = len(set(self._act_dof_arr.tolist())) == len(self._act_dof_arr)
        self._kernel = None
        nat = native.module()
        if (
            nat is not None and not self._balls
        ):  # ball joints need quat_log: stay on the Python path
            self._kernel = nat.Kernel(
                self.model._address, self.data._address, ids.tolist(), self._s_dof.tolist(),
                self._s_qa.tolist(), self._s_da.tolist(), self._act_dof_arr.tolist(), int(reg.n_dof),
            )  # fmt: skip
            self._kernel.set_effort(np.asarray(self._effort_applied, dtype=float))
        self._prepared = True

    def read_state(self, *, contacts: bool, accelerations: bool) -> StateFrame:
        m, d, reg = self.model, self.data, self.reg
        if not getattr(self, "_prepared", False):
            self._prep_read()
        if self._kernel is not None and not (contacts or accelerations):
            tm, pos, quat, lin, ang, q, qd, effort = self._kernel.read()
            return StateFrame(
                t=tm, body_pos=pos, body_quat=quat, body_linvel=lin, body_angvel=ang,
                joint_q=q, joint_qd=qd, joint_effort=effort, ball_slots=self._ball_slots,
            )  # fmt: skip
        if contacts or accelerations:
            mujoco.mj_forward(m, d)
        else:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d)
        ids = self._ids
        nd = len(ids)
        pos, quat = d.xpos[ids], d.xquat[ids]  # fancy indexing copies
        # velocity of each body's reference origin, world frame: cvel is the spatial velocity about the
        # root subtree COM, so v(origin) = v + w x (origin - com). Same arithmetic as mj_objectVelocity.
        cv = d.cvel[ids]
        ang = cv[:, :3].copy()
        lin = cv[:, 3:] + m3.cross_rows(ang, pos - d.subtree_com[self._rootid])
        res = np.zeros(6)
        q, qd = np.zeros(reg.n_dof), np.zeros(reg.n_dof)
        if self._s_dof.size:
            q[self._s_dof] = d.qpos[self._s_qa]
            qd[self._s_dof] = d.qvel[self._s_da]
        for ji in self._balls:
            j = reg.joints[ji]
            qa, da = self._jadr[ji]
            q[j.dof : j.dof + 3] = m3.quat_log(d.qpos[qa : qa + 4])
            qd[j.dof : j.dof + 3] = d.qvel[da : da + 3]
        effort = np.zeros(reg.n_dof)
        if self._act_dof_arr.size:
            if self._act_unique:
                effort[self._act_dof_arr] = self._effort_applied
            else:
                np.add.at(effort, self._act_dof_arr, self._effort_applied)
        linacc = angacc = qdd = None
        if accelerations:
            mujoco.mj_rnePostConstraint(m, d)
            linacc, angacc = np.zeros((nd, 3)), np.zeros((nd, 3))
            for k, mid in enumerate(ids):
                mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_XBODY, int(mid), res, 0)
                angacc[k] = res[:3]
                linacc[k] = res[3:] + self.reg.gravity  # proper -> coordinate acceleration
            qdd = np.zeros(reg.n_dof)
            for ji in reg.dof_joints:
                j = reg.joints[ji]
                qdd[j.dof : j.dof + j.ndof] = d.qacc[self._jadr[ji][1] : self._jadr[ji][1] + j.ndof]
        cs = self._contacts() if contacts else None
        return StateFrame(
            t=float(d.time), body_pos=pos, body_quat=quat, body_linvel=lin, body_angvel=ang,
            joint_q=q, joint_qd=qd, joint_effort=effort, body_linacc=linacc, body_angacc=angacc,
            joint_qdd=qdd, contacts=cs, ball_slots=self._ball_slots,
        )  # fmt: skip

    def _contacts(self) -> ContactSet:
        m, d = self.model, self.data
        n = d.ncon
        if n == 0:
            return ContactSet()
        c = d.contact
        dist, exclude = np.asarray(c.dist), np.asarray(c.exclude)
        idx = np.flatnonzero((dist <= 0) & (exclude == 0))
        if idx.size == 0:
            return ContactSet()
        f6 = np.zeros((idx.size, 6))
        for k, i in enumerate(idx):  # the only per-contact engine call
            mujoco.mj_contactForce(m, d, int(i), f6[k])
        b1 = self._geom_body[np.asarray(c.geom1)[idx]]
        b2 = self._geom_body[np.asarray(c.geom2)[idx]]
        frames = np.asarray(c.frame)[idx].reshape(
            -1, 3, 3
        )  # rows are the contact axes; row 0 = normal
        normal = frames[:, 0, :]
        force_on_2 = np.einsum("nji,nj->ni", frames, f6[:, :3])  # contact frame -> world
        swap = (b1 > b2)[:, None]  # contract: body_a < body_b, normal from A to B
        return ContactSet(
            body_a=np.where(b1 <= b2, b1, b2),
            body_b=np.where(b1 <= b2, b2, b1),
            point=np.asarray(c.pos)[idx].copy(),
            normal=np.where(swap, -normal, normal),
            depth=-dist[idx],
            force_on_b=np.where(swap, -force_on_2, force_on_2),
            normal_force=f6[:, 0].copy(),
        )

    def info(self) -> dict:
        return {
            "engine": "mujoco",
            "version": mujoco.__version__,
            "options": self.opts.model_dump() if hasattr(self.opts, "model_dump") else {},
            "timestep": float(self.model.opt.timestep),
            "approximations": sorted(set(self._approx)),
            "native_kernel": getattr(self, "_kernel", None) is not None,
        }

    @property
    def native(self):
        return self.model, self.data
