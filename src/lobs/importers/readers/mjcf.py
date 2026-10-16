"""MJCF reader (portable mode): walks MuJoCo's *compiled* model, so defaults, includes, frames,
degree/radian conventions and mesh preprocessing are already resolved by MuJoCo itself.

Mapping: bodies -> nodes (parents first); free joints -> a free root; bodies without joints under the
world -> static, under a moving body -> welded; a body with several joints becomes a chain of massless
intermediate bodies; sites -> anchors; position / velocity / motor actuators -> actuators. Tendons,
equality constraints, sensors, plugins and heightfields are not imported (reported).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ... import math3d as m3
from ..ir import IRGeom, IRJoint, IRMaterial, IRNode, IRPhysics, IRWorldPhysics, SceneIR, SourceInfo


def _transform(pos, quat) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = m3.quat_to_mat(np.asarray(quat, float))
    t[:3, 3] = pos
    return t


def read(path, **_):
    import mujoco

    path = Path(path)
    m = mujoco.MjModel.from_xml_path(str(path))
    src = SourceInfo(path=str(path), format="mjcf", meters_per_unit=1.0, up_axis="z")
    ir = SceneIR(src)
    ir.has_physics = True
    ir.world_physics = IRWorldPhysics(
        gravity=tuple(float(x) for x in m.opt.gravity), timestep=float(m.opt.timestep)
    )
    oid = mujoco.mjtObj
    gt = mujoco.mjtGeom

    def name_of(kind, i, prefix):
        n = mujoco.mj_id2name(m, kind, i)
        return n if n else f"{prefix}{i}"

    used: set = set()

    def unique(base: str) -> str:
        n, k = base, 2
        while n in used:
            n, k = f"{base}_{k}", k + 1
        used.add(n)
        return n

    def make_geoms(g: int, mesh_cache: dict):
        t = int(m.geom_type[g])
        size = m.geom_size[g]
        contype, aff = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
        role = "both" if (contype or aff) else "visual"
        tr = _transform(m.geom_pos[g], m.geom_quat[g])
        if t == gt.mjGEOM_BOX:
            return IRGeom(
                kind="box", size=tuple(2 * float(x) for x in size), transform=tr, role=role
            )
        if t == gt.mjGEOM_SPHERE:
            return IRGeom(kind="sphere", radius=float(size[0]), transform=tr, role=role)
        if t in (gt.mjGEOM_CAPSULE, gt.mjGEOM_CYLINDER):
            kind = "capsule" if t == gt.mjGEOM_CAPSULE else "cylinder"
            return IRGeom(
                kind=kind, radius=float(size[0]), length=2 * float(size[1]), transform=tr, role=role
            )
        if t == gt.mjGEOM_PLANE:
            return IRGeom(kind="plane", transform=tr, role=role)
        if t == gt.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            verts = np.asarray(m.mesh_vert[va : va + vn], float)
            faces = np.asarray(m.mesh_face[fa : fa + fn], int)
            tr = tr @ _transform(
                m.mesh_pos[mid], m.mesh_quat[mid]
            )  # mesh frame is recentred by MuJoCo
            return IRGeom(kind="mesh", vertices=verts, faces=faces, transform=tr, role=role)
        ir.warnings.append(f"geom {g} has unsupported type {mujoco.mjtGeom(t).name}; skipped")
        return None

    def fill_geoms(node: IRNode, body: int):
        friction = None
        rgba = None
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != body:
                continue
            geom = make_geoms(g, {})
            if geom is None:
                continue
            node.geoms.append(geom)
            if geom.role != "visual" and friction is None:
                friction = float(m.geom_friction[g][0])
            if rgba is None and np.any(m.geom_rgba[g] != [0.5, 0.5, 0.5, 1.0]):
                rgba = tuple(float(x) for x in m.geom_rgba[g])
        if friction is not None or rgba is not None:
            node.material = IRMaterial(friction=friction, color=rgba, raw={"annotated": False})

    body_node: dict[int, str] = {
        0: ""
    }  # body id -> id of the (final) node that carries its geometry
    joint_ir: dict[int, IRJoint] = {}

    # ---- the world body: each geom is its own static node
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != 0:
            continue
        geom = make_geoms(g, {})
        if geom is None:
            continue
        nm = unique(name_of(oid.mjOBJ_GEOM, g, "geom"))
        tr, geom.transform = geom.transform, np.eye(4)
        node = IRNode(
            id=nm,
            name=nm,
            local=tr,
            geoms=[geom],
            physics=IRPhysics(rigid=False),
            semantic_class="geom",
        )
        if geom.role != "visual":
            node.material = IRMaterial(
                friction=float(m.geom_friction[g][0]), raw={"annotated": False}
            )
        ir.add(node)

    for i in range(1, m.nbody):
        bname = unique(name_of(oid.mjOBJ_BODY, i, "body"))
        parent = int(m.body_parentid[i])
        pnode = body_node.get(parent, "") or None
        jids = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) == i]
        local = _transform(m.body_pos[i], m.body_quat[i])
        types = [int(m.jnt_type[j]) for j in jids]
        chain: list[IRNode] = []
        prev = pnode
        free = mujoco.mjtJoint.mjJNT_FREE in types
        if free and (len(jids) > 1 or pnode is not None):
            ir.warnings.append(
                f"body '{bname}': a free joint must be the only joint of a top-level body; treated as free"
            )
        n_nodes = 1 if (free or not jids) else len(jids)
        for k in range(n_nodes):
            last = k == n_nodes - 1
            nid = bname if last else f"{bname}#j{k}"
            node = IRNode(id=nid, name=bname if last else f"{bname}_j{k}", parent=prev, local=local if k == 0 else np.eye(4),
                          semantic_class="body", tags=[f"group:{name_of(oid.mjOBJ_BODY, p, 'body')}" for p in _ancestors(m, i)])  # fmt: skip
            if last:
                mass = float(m.body_mass[i])
                if mass > 0:
                    r = m3.quat_to_mat(np.asarray(m.body_iquat[i], float))
                    node.physics = IRPhysics(mass=mass, com=tuple(float(x) for x in m.body_ipos[i]),
                                             inertia=r @ np.diag(np.asarray(m.body_inertia[i], float)) @ r.T)  # fmt: skip
                fill_geoms(node, i)
                for s in range(m.nsite):
                    if int(m.site_bodyid[s]) == i:
                        node.sites.append((name_of(oid.mjOBJ_SITE, s, "site"), tuple(float(x) for x in m.site_pos[s]),
                                           tuple(float(x) for x in m.site_quat[s])))  # fmt: skip
            else:  # massless intermediate body of a multi-joint chain
                node.physics = IRPhysics(mass=1e-4, com=(0.0, 0.0, 0.0), inertia=np.eye(3) * 1e-9)
                node.properties["synthetic"] = True
            node.physics = node.physics or IRPhysics()
            if free:
                node.physics.rigid = True
                node.parent = None
                node.local = local
            elif not jids:
                node.physics.rigid = False if pnode is None else None
            ir.add(node)
            chain.append(node)
            if jids and not free:
                jid = jids[k]
                ir_joint = _make_joint(
                    m, mujoco, jid, prev, node.id, name_of(oid.mjOBJ_JOINT, jid, "joint")
                )
                ir.joints.append(ir_joint)
                joint_ir[jid] = ir_joint
            prev = node.id
        if not jids and pnode is not None:  # welded to a moving parent
            ir.joints.append(
                IRJoint(
                    id=f"{bname}#weld",
                    name=f"{bname}_weld",
                    type="fixed",
                    parent_node=pnode,
                    child_node=bname,
                )
            )
        body_node[i] = chain[-1].id

    # ---- actuators
    for a in range(m.nu):
        if int(m.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            ir.warnings.append(
                f"actuator '{name_of(oid.mjOBJ_ACTUATOR, a, 'actuator')}': only joint transmissions are imported"
            )
            continue
        jid = int(m.actuator_trnid[a][0])
        j = joint_ir.get(jid)
        if j is None or j.type not in ("revolute", "continuous", "prismatic"):
            continue
        gain, bias, gear = (
            m.actuator_gainprm[a],
            m.actuator_biasprm[a],
            float(m.actuator_gear[a][0]),
        )
        ctrl = (
            tuple(float(x) for x in m.actuator_ctrlrange[a])
            if int(m.actuator_ctrllimited[a])
            else None
        )
        frc = float(m.actuator_forcerange[a][1]) if int(m.actuator_forcelimited[a]) else None
        if int(m.actuator_biastype[a]) == 0 and int(m.actuator_gaintype[a]) == 0:  # motor
            scale = gear * float(gain[0])
            j.actuator = {
                "kind": "effort",
                "ctrl_range": None if ctrl is None else (ctrl[0] * scale, ctrl[1] * scale),
                "effort_limit": frc,
            }
        elif (
            int(m.actuator_biastype[a]) == 1
            and abs(float(bias[1]) + float(gain[0])) < 1e-9
            and gear == 1.0
        ):
            j.actuator = {
                "kind": "position",
                "kp": float(gain[0]),
                "kd": -float(bias[2]),
                "ctrl_range": ctrl,
                "effort_limit": frc,
            }
        elif (
            int(m.actuator_biastype[a]) == 1
            and abs(float(bias[2]) + float(gain[0])) < 1e-9
            and float(bias[1]) == 0.0
            and gear == 1.0
        ):
            j.actuator = {
                "kind": "velocity",
                "kd": float(gain[0]),
                "ctrl_range": ctrl,
                "effort_limit": frc,
            }
        else:
            ir.warnings.append(
                f"actuator '{name_of(oid.mjOBJ_ACTUATOR, a, 'actuator')}': unsupported gain/bias/gear; not imported"
            )

    for count, what in ((m.ntendon, "tendons"), (m.neq, "equality constraints"), (m.nsensor, "sensors"), (m.nhfield, "height fields"),
                        (m.nplugin, "plugins"), (m.nmocap, "mocap bodies")):  # fmt: skip
        if count:
            ir.warnings.append(f"not imported: {count} {what}")
    return ir


def _ancestors(m, body: int) -> list[int]:
    out, p = [], int(m.body_parentid[body])
    while p > 0:
        out.append(p)
        p = int(m.body_parentid[p])
    return list(reversed(out))


def _make_joint(m, mujoco, jid: int, parent: str | None, child: str, name: str) -> IRJoint:
    t = int(m.jnt_type[jid])
    dof = int(m.jnt_dofadr[jid])
    kw = dict(damping=float(m.dof_damping[dof]), armature=float(m.dof_armature[dof]))
    pos = tuple(float(x) for x in m.jnt_pos[jid])
    if t == mujoco.mjtJoint.mjJNT_BALL:
        return IRJoint(
            id=f"{child}/{name}",
            name=name,
            type="spherical",
            parent_node=parent,
            child_node=child,
            pos=pos,
            **kw,
        )
    limited = bool(m.jnt_limited[jid])
    rng = (
        (float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])) if limited else None
    )  # radians / metres
    axis = tuple(float(x) for x in m.jnt_axis[jid])
    if t == mujoco.mjtJoint.mjJNT_HINGE:
        return IRJoint(id=f"{child}/{name}", name=name, type="revolute" if limited else "continuous", parent_node=parent,
                       child_node=child, pos=pos, axis=axis, limits=rng, **kw)  # fmt: skip
    return IRJoint(
        id=f"{child}/{name}",
        name=name,
        type="prismatic",
        parent_node=parent,
        child_node=child,
        pos=pos,
        axis=axis,
        limits=rng,
        **kw,
    )
