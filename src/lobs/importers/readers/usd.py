"""OpenUSD + UsdPhysics reader (usd-core).

Semantics handled explicitly (each is a classic way to get a USD scene subtly wrong):
  * stage units (metersPerUnit) and up axis (Y or Z) -> reported to normalization;
  * gravity magnitude is in stage units / s^2 (converted to m/s^2 here);
  * revolute limits are in DEGREES, prismatic limits in stage units; angular drive gains are per degree;
  * rigid-body ownership: a collider belongs to its nearest rigid-body ancestor, otherwise it is static;
  * in a stage that has physics, geometry without a CollisionAPI is visual only;
  * joints to the world (an unset body0/body1) are represented with parent/child = None.
"""

from __future__ import annotations

import numpy as np

from ..ir import (
    IRGeom,
    IRJoint,
    IRMaterial,
    IRNode,
    IRPhysics,
    IRWorldPhysics,
    SceneIR,
    SourceInfo,
    apply_lobs_annotations,
)

_APPROX = {
    "none": "mesh", "meshSimplification": "mesh", "convexHull": "convex_hull",
    "convexDecomposition": "decompose", "boundingCube": "box_fit", "boundingSphere": "convex_hull",
    "sdf": "mesh",
}  # fmt: skip
_AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _mat(m) -> np.ndarray:
    """Gf.Matrix4d (row-vector convention) -> numpy column-vector 4x4."""
    return np.array(m, dtype=float).T


def _rigid(world: np.ndarray):
    """Split a world matrix into a rigid frame and its per-axis scale (shear is folded into the scale)."""
    lin = world[:3, :3]
    scale = np.linalg.norm(lin, axis=0)
    scale[scale == 0] = 1.0
    rot = lin / scale
    u, _, vt = np.linalg.svd(rot)  # re-orthonormalize
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot[:, 0] = -rot[:, 0]
        scale[0] = -scale[0]
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, world[:3, 3]
    return t, scale


def _quat_to_mat(q) -> np.ndarray:
    w, (x, y, z) = float(q.GetReal()), [float(c) for c in q.GetImaginary()]
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def _axis_to_z(axis: str) -> np.ndarray:
    """Rotation taking USD primitive axis X/Y/Z onto +Z (the LoBS primitive axis)."""
    if axis == "Z":
        return np.eye(3)
    if axis == "X":
        return np.array(
            [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        )  # rotate about Y by -90: x -> z
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )  # about X by +90... maps y -> z


def _get(attr, default=None):
    if attr is None or not attr.IsValid():
        return default
    v = attr.Get()
    return default if v is None else v


def read(path, **_):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade  # noqa: F401

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"cannot open USD stage {path}")
    mpu = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    try:
        kpu = float(UsdPhysics.GetStageKilogramsPerUnit(stage) or 1.0)
    except Exception:  # noqa: BLE001 - older USD builds
        kpu = 1.0
    dens_si = kpu / mpu**3  # USD density is mass-unit per length-unit^3 -> kg/m^3
    up = str(UsdGeom.GetStageUpAxis(stage) or "Y").lower()
    src = SourceInfo(path=str(path), format="usd", meters_per_unit=mpu, up_axis=up)
    ir = SceneIR(src)
    tc = Usd.TimeCode.Default()
    cache = UsdGeom.XformCache(tc)

    prims = [
        p
        for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
        if p.IsActive() and p.IsDefined()
    ]
    rigid_paths = set()
    has_physics = False
    for p in prims:
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_paths.add(p.GetPath())
            has_physics = True
        if p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdPhysics.Scene):
            has_physics = True
    ir.has_physics = has_physics

    # ---- world physics
    for p in prims:
        if p.IsA(UsdPhysics.Scene):
            sc = UsdPhysics.Scene(p)
            d = _get(sc.GetGravityDirectionAttr(), Gf.Vec3f(0, 0, 0))
            mag = _get(sc.GetGravityMagnitudeAttr(), float("-inf"))
            d = np.array([d[0], d[1], d[2]], dtype=float)
            if not np.any(d):  # "use the negative up axis"
                d = -np.array(_AXIS_VEC[up.upper()])
            d = d / np.linalg.norm(d)
            g = (
                9.81 if (mag is None or not np.isfinite(mag) or mag < 0) else float(mag) * mpu
            )  # stage units/s^2 -> m/s^2
            ir.world_physics = IRWorldPhysics(gravity=tuple(float(x) for x in d * g))
            break

    def owner_of(prim):
        cur = prim
        while cur and cur.GetPath() != Sdf.Path.absoluteRootPath:
            if cur.GetPath() in rigid_paths:
                return cur
            cur = cur.GetParent()
        return None

    def is_collider(prim):
        cur = prim
        while cur and cur.GetPath() != Sdf.Path.absoluteRootPath:
            if cur.HasAPI(UsdPhysics.CollisionAPI):
                return cur
            cur = cur.GetParent()
        return None

    gprim_types = (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Cylinder,
        UsdGeom.Capsule,
        UsdGeom.Plane,
    )

    def tags_for(prim):
        names, cur = [], prim.GetParent()
        while cur and cur.GetPath() != Sdf.Path.absoluteRootPath:
            names.append(cur.GetName())
            cur = cur.GetParent()
        return [f"group:{n}" for n in reversed(names)]

    def physics_material(prim):
        try:
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                materialPurpose="physics"
            )
        except Exception:  # noqa: BLE001
            mat = None
        if not mat or not mat.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
            return None
        api = UsdPhysics.MaterialAPI(mat.GetPrim())
        fr = _get(api.GetDynamicFrictionAttr(), None)
        if fr is None:
            fr = _get(api.GetStaticFrictionAttr(), None)
        dens = _get(api.GetDensityAttr(), 0.0)
        return IRMaterial(name=mat.GetPrim().GetName(), density=float(dens) * dens_si if dens and dens > 0 else None,
                          friction=None if fr is None else float(fr), restitution=_get(api.GetRestitutionAttr(), None),
                          raw={"annotated": False})  # fmt: skip

    def make_geom(prim, body_world_rigid, body_world_full) -> IRGeom | None:
        """IRGeom of one gprim expressed in the body's rigid frame (scale baked in)."""
        if UsdGeom.Imageable(prim).GetPurposeAttr().Get() == "guide":
            return None
        w = _mat(cache.GetLocalToWorldTransform(prim))
        rel = (
            np.linalg.inv(body_world_rigid) @ w
        )  # prim frame (with its scale) in the body rigid frame
        lin, tr = rel[:3, :3], rel[:3, 3]
        rigid_part, sc = _rigid(rel)
        if prim.IsA(UsdGeom.Mesh):
            m = UsdGeom.Mesh(prim)
            pts = np.array(_get(m.GetPointsAttr(), []), dtype=float).reshape(-1, 3)
            counts = np.array(_get(m.GetFaceVertexCountsAttr(), []), dtype=int)
            idx = np.array(_get(m.GetFaceVertexIndicesAttr(), []), dtype=int)
            if len(pts) == 0 or len(counts) == 0:
                return None
            faces, o = [], 0
            for c in counts:
                for k in range(1, c - 1):
                    faces.append((idx[o], idx[o + k], idx[o + k + 1]))
                o += c
            verts = pts @ lin.T + tr
            if np.linalg.det(lin) < 0:  # mirrored: keep outward winding
                faces = [(a, c, b) for a, b, c in faces]
            return IRGeom(kind="mesh", vertices=verts, faces=np.array(faces, dtype=int))
        if prim.IsA(UsdGeom.Cube):
            size = float(_get(UsdGeom.Cube(prim).GetSizeAttr(), 2.0))
            return IRGeom(kind="box", size=tuple(abs(size * s) for s in sc), transform=rigid_part)
        if prim.IsA(UsdGeom.Sphere):
            r = float(_get(UsdGeom.Sphere(prim).GetRadiusAttr(), 1.0))
            if np.ptp(np.abs(sc)) > 1e-6 * max(abs(sc).max(), 1):
                ir.warnings.append(
                    f"{prim.GetPath()}: non-uniform scaled sphere approximated by its mean radius"
                )
            return IRGeom(
                kind="sphere", radius=r * float(np.mean(np.abs(sc))), transform=rigid_part
            )
        if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            cyl = prim.IsA(UsdGeom.Cylinder)
            api = UsdGeom.Cylinder(prim) if cyl else UsdGeom.Capsule(prim)
            axis = str(_get(api.GetAxisAttr(), "Z"))
            r, h = float(_get(api.GetRadiusAttr(), 0.5)), float(_get(api.GetHeightAttr(), 2.0))
            ai = "XYZ".index(axis)
            along = abs(sc[ai])
            across = float(np.mean([abs(sc[i]) for i in range(3) if i != ai]))
            t = rigid_part.copy()
            t[:3, :3] = rigid_part[:3, :3] @ _axis_to_z(axis).T  # LoBS z axis = USD axis
            return IRGeom(
                kind="cylinder" if cyl else "capsule",
                radius=r * across,
                length=h * along,
                transform=t,
            )
        if prim.IsA(UsdGeom.Plane):
            axis = str(_get(UsdGeom.Plane(prim).GetAxisAttr(), "Z"))
            t = rigid_part.copy()
            t[:3, :3] = rigid_part[:3, :3] @ _axis_to_z(axis).T
            return IRGeom(kind="plane", transform=t)
        return None

    nodes_by_body: dict = {}

    def add_body_node(prim, rigid_world, full_world):
        nid = str(prim.GetPath())
        node = IRNode(
            id=nid,
            name=prim.GetName(),
            local=rigid_world,
            semantic_class=prim.GetTypeName() or None,
            tags=tags_for(prim),
        )
        nodes_by_body[nid] = node
        return node

    # ---- rigid bodies first (so their order is the traversal order)
    body_world: dict = {}
    for p in prims:
        if p.GetPath() in rigid_paths:
            full = _mat(cache.GetLocalToWorldTransform(p))
            rigid_world, _scale = _rigid(full)
            node = add_body_node(p, rigid_world, full)
            body_world[str(p.GetPath())] = (rigid_world, _scale)
            rb = UsdPhysics.RigidBodyAPI(p)
            enabled = bool(_get(rb.GetRigidBodyEnabledAttr(), True))
            kinematic = bool(_get(rb.GetKinematicEnabledAttr(), False))
            node.physics = IRPhysics(rigid=enabled and not kinematic)
            if p.HasAPI(UsdPhysics.MassAPI):
                m = UsdPhysics.MassAPI(p)
                mass = _get(m.GetMassAttr(), 0.0)
                if mass and mass > 0:
                    node.physics.mass = float(mass) * kpu
                com = _get(m.GetCenterOfMassAttr(), None)
                if com is not None and np.all(np.isfinite(list(com))):
                    node.physics.com = tuple(
                        float(c) * s
                        for c, s in zip(com, body_world[str(p.GetPath())][1], strict=True)
                    )
                di = _get(m.GetDiagonalInertiaAttr(), None)
                if di is not None and np.all(np.asarray(di) > 0):
                    pa = _quat_to_mat(_get(m.GetPrincipalAxesAttr(), Gf.Quatf(1, 0, 0, 0)))
                    node.physics.inertia = (
                        (pa @ np.diag([float(x) for x in di]) @ pa.T) * kpu * mpu**2
                    )  # kg*unit^2 -> kg*m^2
                dens = _get(m.GetDensityAttr(), 0.0)
                if dens and dens > 0:
                    node.material = IRMaterial(
                        density=float(dens) * dens_si, raw={"annotated": False}
                    )
            apply_lobs_annotations(
                node,
                {
                    a.GetName(): a.Get()
                    for a in p.GetAttributes()
                    if a.GetName().startswith("lobs:")
                },
            )

    n_static = 0
    for p in prims:
        if not any(p.IsA(t) for t in gprim_types):
            continue
        owner = owner_of(p)
        collider_prim = is_collider(p)
        role = "both" if (collider_prim is not None or not has_physics) else "visual"
        if owner is not None:
            rw, _ = body_world[str(owner.GetPath())]
            geom = make_geom(p, rw, None)
            if geom is None:
                continue
            geom.role = role
            geom.material = physics_material(p) if collider_prim is not None else None
            node = nodes_by_body[str(owner.GetPath())]
            node.geoms.append(geom)
            if node.material is None and geom.material is not None:
                node.material = geom.material
            elif (
                geom.material is not None
                and node.material is not None
                and node.material.name is None
            ):
                node.material = IRMaterial(name=geom.material.name, density=node.material.density, friction=geom.material.friction,
                                           restitution=geom.material.restitution, raw={"annotated": False})  # fmt: skip
            if collider_prim is not None and p.IsA(UsdGeom.Mesh):
                hint = _approximation(collider_prim, p, UsdPhysics)
                if hint:
                    node.physics.collision_hint = hint
            if collider_prim is not None and not bool(
                _get(UsdPhysics.CollisionAPI(collider_prim).GetCollisionEnabledAttr(), True)
            ):
                geom.role = "visual"
            continue
        # ---- static geometry: one node per gprim
        full = _mat(cache.GetLocalToWorldTransform(p))
        rigid_world, _ = _rigid(full)
        geom = make_geom(p, rigid_world, None)
        if geom is None:
            continue
        geom.role = role
        geom.material = physics_material(p) if collider_prim is not None else None
        node = IRNode(id=str(p.GetPath()), name=p.GetName(), local=rigid_world, semantic_class=p.GetTypeName() or None,
                      tags=tags_for(p), geoms=[geom], material=geom.material)  # fmt: skip
        node.physics = IRPhysics(
            rigid=False, collision_enabled=(collider_prim is not None) if has_physics else None
        )
        if collider_prim is not None and p.IsA(UsdGeom.Mesh):
            hint = _approximation(collider_prim, p, UsdPhysics)
            if hint:
                node.physics.collision_hint = hint
        apply_lobs_annotations(
            node,
            {a.GetName(): a.Get() for a in p.GetAttributes() if a.GetName().startswith("lobs:")},
        )
        ir.add(node)
        n_static += 1

    for node in nodes_by_body.values():
        ir.add(node)

    # ---- joints
    for p in prims:
        if not p.IsA(UsdPhysics.Joint):
            continue
        jt = _joint_type(p, UsdPhysics)
        j = UsdPhysics.Joint(p)
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        b0 = b0[0] if b0 and b0[0] in ir.nodes else None
        b1 = b1[0] if b1 and b1[0] in ir.nodes else None
        if b0 is None and b1 is None:
            ir.warnings.append(f"joint {p.GetPath()} connects nothing known; ignored")
            continue
        flip = b1 is None  # only body0 given: it becomes the child
        child, parent = (b0, None) if flip else (b1, b0)
        if child not in body_world:
            ir.warnings.append(f"joint {p.GetPath()} is attached to a static body; ignored")
            continue
        lp = np.array(
            list(_get(j.GetLocalPos0Attr() if flip else j.GetLocalPos1Attr(), (0, 0, 0))),
            dtype=float,
        )
        lr = _quat_to_mat(
            _get(j.GetLocalRot0Attr() if flip else j.GetLocalRot1Attr(), _identity_quat())
        )
        scale = body_world[child][1]
        pos_child = lp * scale
        axis_name = "X"
        limits = None
        drive = None
        eff = None
        if jt in ("revolute", "prismatic"):
            api = UsdPhysics.RevoluteJoint(p) if jt == "revolute" else UsdPhysics.PrismaticJoint(p)
            axis_name = str(_get(api.GetAxisAttr(), "X"))
            lo, hi = _get(api.GetLowerLimitAttr(), None), _get(api.GetUpperLimitAttr(), None)
            if (
                lo is not None
                and hi is not None
                and np.isfinite(lo)
                and np.isfinite(hi)
                and lo < hi
            ):
                limits = (
                    (np.radians(lo), np.radians(hi)) if jt == "revolute" else (float(lo), float(hi))
                )  # revolute is DEGREES
            drive_api = UsdPhysics.DriveAPI.Get(p, "angular" if jt == "revolute" else "linear")
            if drive_api and drive_api.GetStiffnessAttr().IsValid():
                kp, kd = (
                    float(_get(drive_api.GetStiffnessAttr(), 0.0)),
                    float(_get(drive_api.GetDampingAttr(), 0.0)),
                )
                if kp > 0 or kd > 0:
                    k = (
                        180.0 / np.pi if jt == "revolute" else 1.0 / mpu
                    )  # per degree -> per radian ; per stage unit -> per metre
                    drive = (kp * k, kd * k)
                mf = _get(drive_api.GetMaxForceAttr(), None)
                if mf is not None and np.isfinite(mf) and mf > 0:
                    eff = float(mf)
        axis_child = lr @ np.array(_AXIS_VEC[axis_name])
        ir.joints.append(IRJoint(id=str(p.GetPath()), name=p.GetName(), type=jt, parent_node=parent, child_node=child,
                                 pos=tuple(float(x) for x in pos_child), axis=tuple(float(x) for x in axis_child),
                                 limits=limits, drive=drive, effort_limit=eff))  # fmt: skip
    if not ir.nodes:
        ir.warnings.append("no geometry found in the stage")
    return ir


def _identity_quat():
    from pxr import Gf

    return Gf.Quatf(1, 0, 0, 0)


def _joint_type(prim, UsdPhysics) -> str:
    if prim.IsA(UsdPhysics.RevoluteJoint):
        return "revolute"
    if prim.IsA(UsdPhysics.PrismaticJoint):
        return "prismatic"
    if prim.IsA(UsdPhysics.FixedJoint):
        return "fixed"
    if prim.IsA(UsdPhysics.SphericalJoint):
        return "spherical"
    return "planar"  # generic / distance joints: reported as unsupported by physicalize


def _approximation(collider_prim, mesh_prim, UsdPhysics):
    for prim in (mesh_prim, collider_prim):
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            tok = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if tok:
                return _APPROX.get(str(tok))
    return None
