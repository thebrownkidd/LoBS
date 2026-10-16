import numpy as np
import pytest

import lobs

from .fixtures.make_usd import make_usd_scene

BACKENDS = lobs.backends.available()


def src(world, path):
    return next(a for a in world.spec.actors if a.source and a.source.source_id == path)


@pytest.fixture(scope="module")
def usd(tmp_path_factory):
    return make_usd_scene(tmp_path_factory.mktemp("usd") / "scene.usda")


@pytest.fixture(scope="module")
def imported(usd, tmp_path_factory):
    # the stage declares all physics: the motion mode only matters for things the stage does not say
    return lobs.import_world(usd, motion="auto", out=tmp_path_factory.mktemp("o") / "u.lobs")


def test_stage_units_up_axis_and_gravity(imported):
    world, report = imported
    d = {(x.target, x.field): x for x in report.decisions.items}
    assert d[("scene", "units")].provenance == "source" and d[("scene", "units")].value.startswith(
        "0.01"
    )
    assert d[("scene", "up_axis")].value == "y" and d[("scene", "up_axis")].provenance == "source"
    # 981 cm/s^2 pointing -Y in a Y-up stage -> 9.81 m/s^2 along -Z after conversion
    assert world.spec.gravity == pytest.approx((0.0, 0.0, -9.81), abs=1e-6)


def test_rigid_bodies_static_colliders_and_visual_only(imported):
    world, report = imported
    crate, ground = src(world, "/World/Crate"), src(world, "/World/Ground")
    assert crate.root == "free" and ground.root == "fixed"
    assert (
        next(
            d for d in report.decisions.items if d.target == "/World/Crate" and d.field == "motion"
        ).provenance
        == "source"
    )
    # crate: 5 kg stated by MassAPI, collider is a 0.4 m cube, the sphere marker has no collision (visual only)
    assert crate.bodies[0].inertial.mass == pytest.approx(5.0)
    assert crate.bodies[0].inertial.mode == "explicit"
    coll = [g for g in crate.bodies[0].geoms if g.role != "visual"]
    assert (
        len(coll) == 1
        and coll[0].shape.type == "box"
        and coll[0].shape.size == pytest.approx((0.4, 0.4, 0.4))
    )
    spheres = [g for g in crate.bodies[0].geoms if g.shape.type == "sphere"]
    assert spheres and all(g.role == "visual" for g in spheres)  # kept for viewers, never collides
    # ground: 20 x 0.2 x 20 m slab, top at z=0 after Y->Z (centre 0.1 below)
    assert ground.pose.pos[2] == pytest.approx(-0.1, abs=1e-9)
    gbox = next(g for g in ground.bodies[0].geoms if g.role != "visual")
    assert sorted(gbox.shape.size) == pytest.approx([0.2, 20.0, 20.0], abs=1e-6)
    # wedge: density from MassAPI (1000 kg/m^3) x hull volume; MeshCollisionAPI(convexHull) is a source hint
    wedge = src(world, "/World/Wedge")
    assert (
        next(
            d
            for d in report.decisions.items
            if d.target == "/World/Wedge" and d.field == "collision"
        ).provenance
        == "source"
    )
    assert any(g.shape.type == "convex_mesh" for g in wedge.bodies[0].geoms)
    import trimesh

    pts = [
        (-20, -10, -20),
        (20, -10, -20),
        (20, -10, 20),
        (-20, -10, 20),
        (-20, 10, -20),
        (20, 10, -20),
    ]  # cm
    volume_m3 = trimesh.convex.convex_hull(pts).volume * 1e-6
    assert wedge.bodies[0].inertial.mass == pytest.approx(
        volume_m3 * 1000.0, rel=1e-3
    )  # MassAPI density 1000 kg/m^3


def test_physics_material_friction_and_restitution(imported):
    world, report = imported
    crate = src(world, "/World/Crate")
    mat = next(g.material for g in crate.bodies[0].geoms if g.role != "visual")
    assert mat.friction == pytest.approx(0.9) and mat.restitution == pytest.approx(0.2)


def test_joint_to_world_limits_in_degrees(imported):
    world, report = imported
    rod = src(world, "/World/Rod")
    assert len(rod.bodies) == 2 and rod.root == "fixed"  # synthetic fixed world base + the rod
    j = rod.joints[0]
    assert j.type == "hinge"
    assert j.range == pytest.approx(
        (np.radians(-30), np.radians(60)), abs=1e-9
    )  # degrees -> radians
    # joint position is the rod top: 0.5 m up its own Y axis (-> Z after the axis change)
    assert np.linalg.norm(j.pos) == pytest.approx(0.5, abs=1e-6)
    assert np.linalg.norm(j.axis) == pytest.approx(1.0, abs=1e-9)
    reg = world.registry()
    assert any(jj.path.endswith(j.name) and jj.range is not None for jj in reg.joints)


@pytest.mark.parametrize("be", BACKENDS)
def test_usd_scene_simulates_to_expected_rest_heights_and_limits(imported, be):
    world, _ = imported
    rod = src(world, "/World/Rod")
    names = {a.name for a in world.spec.actors}
    assert {"Crate", "Rod"} <= names
    for m in ("crate_h", "q"):
        world.spec.measures.pop(m, None)
    world.measure("crate_h", lobs.props.height(world.actor(src(world, "/World/Crate").name)))
    world.measure("q", lobs.props.joint_position(world.actor(rod.name).joint(rod.joints[0].name)))
    report = world.validate(be)
    if not report.ok:
        pytest.skip(report.format())
    res = lobs.check_world(world, be, seconds=1.5)
    assert res.passed, res.summary()
    sim = world.simulate(be)
    sim.run(seconds=2.0)
    assert sim["crate_h"] == pytest.approx(0.2, abs=1.5e-2)  # 0.4 m cube on the ground
    q = sim.history["q"]
    assert (
        q.min() >= np.radians(-30) - 0.06 and q.max() <= np.radians(60) + 0.06
    )  # stays within the stage's limits
