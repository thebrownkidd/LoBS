"""A small UsdPhysics stage, authored with pxr: centimetres, Y-up, joints in degrees."""

from __future__ import annotations

from pathlib import Path


def make_usd_scene(path: Path) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # centimetres
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Xform.Define(stage, "/World")
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, -1.0, 0.0))
    scene.CreateGravityMagnitudeAttr(981.0)  # stage units (cm) per s^2

    # static ground: a collider with no rigid body. Cube of size 1 scaled to 20 m x 20 cm x 20 m, top at y=0
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0, -10, 0))
    ground.AddScaleOp().Set(Gf.Vec3f(2000, 20, 2000))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # physics material
    mat = UsdShade.Material.Define(stage, "/World/Materials/Rubber")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(0.9)
    mapi.CreateDynamicFrictionAttr(0.9)
    mapi.CreateRestitutionAttr(0.2)

    # dynamic crate: 40 cm cube collider, 5 kg, rubber, dropped from 0.2 m above its resting height
    crate = UsdGeom.Xform.Define(stage, "/World/Crate")
    crate.AddTranslateOp().Set(Gf.Vec3d(100, 40, 0))
    UsdPhysics.RigidBodyAPI.Apply(crate.GetPrim())
    UsdPhysics.MassAPI.Apply(crate.GetPrim()).CreateMassAttr(5.0)
    col = UsdGeom.Cube.Define(stage, "/World/Crate/Collider")
    col.CreateSizeAttr(40.0)
    UsdPhysics.CollisionAPI.Apply(col.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics"
    )
    vis = UsdGeom.Sphere.Define(
        stage, "/World/Crate/Marker"
    )  # visual only: no CollisionAPI in a physics stage
    vis.CreateRadiusAttr(5.0)

    # dynamic wedge with a convex-hull mesh collider
    wedge = UsdGeom.Xform.Define(stage, "/World/Wedge")
    wedge.AddTranslateOp().Set(Gf.Vec3d(-100, 30, 0))
    UsdPhysics.RigidBodyAPI.Apply(wedge.GetPrim())
    UsdPhysics.MassAPI.Apply(wedge.GetPrim()).CreateDensityAttr(0.001)  # kg/cm^3 -> 1000 kg/m^3
    mesh = UsdGeom.Mesh.Define(stage, "/World/Wedge/Mesh")
    pts = [
        (-20, -10, -20),
        (20, -10, -20),
        (20, -10, 20),
        (-20, -10, 20),
        (-20, 10, -20),
        (20, 10, -20),
    ]
    import trimesh

    hull = trimesh.convex.convex_hull(pts)  # outward-wound triangles
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float, v)) for v in hull.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * len(hull.faces))
    mesh.CreateFaceVertexIndicesAttr([int(i) for i in hull.faces.ravel()])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")

    # pendulum: a rod hinged to the WORLD (body0 unset) about the X axis, limits -30..60 degrees
    rod = UsdGeom.Xform.Define(stage, "/World/Rod")
    rod.AddTranslateOp().Set(
        Gf.Vec3d(0, 250, -150)
    )  # rod centre; hangs 100 cm long, top end at y=300
    UsdPhysics.RigidBodyAPI.Apply(rod.GetPrim())
    UsdPhysics.MassAPI.Apply(rod.GetPrim()).CreateMassAttr(2.0)
    cap = UsdGeom.Capsule.Define(stage, "/World/Rod/Collider")
    cap.CreateAxisAttr("Y")
    cap.CreateRadiusAttr(4.0)
    cap.CreateHeightAttr(92.0)
    UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge")
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/World/Rod")])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0, 300, -150))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 50, 0))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    hinge.CreateAxisAttr("X")
    hinge.CreateLowerLimitAttr(-30.0)
    hinge.CreateUpperLimitAttr(60.0)

    stage.GetRootLayer().Save()
    return path
