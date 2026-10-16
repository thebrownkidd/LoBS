import numpy as np
import pytest

import lobs
from lobs import math3d as m3
from lobs import props
from lobs.errors import ImportFailure

from .fixtures.make_robot import make_urdf

BACKENDS = lobs.backends.available()


@pytest.fixture(scope="module")
def urdf(tmp_path_factory):
    return make_urdf(tmp_path_factory.mktemp("urdf"))


@pytest.fixture(scope="module")
def imported(urdf, tmp_path_factory):
    path, kw = urdf
    return lobs.import_world(
        path, motion="auto", profile="robotics", out=tmp_path_factory.mktemp("o") / "arm.lobs", **kw
    )


def body(world, name):
    return next(b for a in world.spec.actors for b in a.bodies if b.name == name)


def test_tree_joints_limits_and_frames(imported):
    world, report = imported
    arm = next(
        a for a in world.spec.actors if a.name == "base" or a.name == "arm" or len(a.bodies) > 1
    )
    assert [b.name for b in arm.bodies] == [
        "base",
        "upper",
        "ghost",
        "slider",
        "wheel",
    ]  # parents first (BFS)
    j = {x.name: x for x in arm.joints}
    assert j["shoulder"].type == "hinge" and j["shoulder"].range == pytest.approx((-1.0, 1.2))
    assert j["slide"].type == "slide" and j["slide"].range == pytest.approx((-0.1, 0.2))
    assert j["spin"].type == "hinge" and j["spin"].range is None  # continuous
    assert j["mount"].type == "fixed"
    assert j["shoulder"].damping == pytest.approx(0.01) and j["shoulder"].axis == pytest.approx(
        (1, 0, 0)
    )
    # rpy convention: yaw of pi/2 about z, translated 0.1 up
    up = body(world, "upper")
    assert up.pose.pos == pytest.approx((0, 0, 0.1), abs=1e-9)
    assert m3.quat_same_rotation(up.pose.quat, m3.quat_from_axis_angle([0, 0, 1], np.pi / 2), 1e-6)
    assert arm.root == "fixed"  # fixed_base=True is the default for robots


def test_inertial_com_and_rotated_tensor(imported):
    world, _ = imported
    wheel = body(world, "wheel").inertial
    assert wheel.mass == pytest.approx(0.5) and wheel.com == pytest.approx((0, 0.01, 0))
    r = m3.quat_to_mat(
        m3.quat_from_axis_angle([0, 0, 1], 0.5)
    )  # the inertial frame has rpy = (0, 0, 0.5)
    tensor = np.array([[0.005, 0.001, 0], [0.001, 0.004, 0], [0, 0, 0.003]])
    assert np.asarray(wheel.inertia) == pytest.approx(r @ tensor @ r.T, abs=1e-9)
    assert body(world, "upper").inertial.mass == pytest.approx(2.0)
    # the root of a fixed-base robot is welded to the world: it carries no dynamics
    assert body(world, "base").inertial.mass == 0.0


def test_mesh_via_package_uri_and_geometry_roles(imported):
    world, report = imported
    slider = body(world, "slider")
    shapes = {(g.shape.type, g.role) for g in slider.geoms}
    assert ("triangle_mesh", "visual") in shapes and (
        "box",
        "collision",
    ) in shapes  # cube fitted exactly
    assert ("capsule", "collision") in {(g.shape.type, g.role) for g in body(world, "upper").geoms}
    assert ("cylinder", "visual") in {(g.shape.type, g.role) for g in body(world, "upper").geoms}
    assert world.base_dir is not None and any((world.base_dir / "assets").iterdir())
    # links without inertial / geometry get a placeholder mass instead of a singular body
    ghost = body(world, "ghost").inertial
    assert 0 < ghost.mass < 0.01
    assert any(d.target == "ghost" and d.review for d in report.decisions.items)


def test_floating_base_and_missing_mesh_warning(urdf, tmp_path):
    path, kw = urdf
    world, _ = lobs.import_world(
        path, motion="auto", profile="robotics", out=tmp_path / "f.lobs", fixed_base=False, **kw
    )
    assert next(a for a in world.spec.actors if len(a.bodies) > 1).root == "free"
    bad, kw2 = make_urdf(tmp_path / "nomesh", with_mesh=False)
    world, report = lobs.import_world(
        bad, motion="auto", profile="robotics", out=tmp_path / "n.lobs", **kw2
    )
    assert any("not found" in w and "block.stl" in w for w in report.warnings)
    assert world.validate().ok


def test_continuous_floating_planar_policy(tmp_path):
    p = tmp_path / "x.urdf"
    p.write_text('<robot name="x"><link name="a"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>'
                 '<collision><geometry><sphere radius="0.1"/></geometry></collision></link>'
                 '<link name="b"><inertial><mass value="1"/><inertia ixx="1" iyy="1" izz="1"/></inertial>'
                 '<collision><geometry><sphere radius="0.1"/></geometry></collision></link>'
                 '<joint name="j" type="planar"><parent link="a"/><child link="b"/></joint></robot>')  # fmt: skip
    with pytest.raises(ImportFailure, match="planar"):
        lobs.import_world(p, motion="auto", profile="robotics", out=tmp_path / "e.lobs")
    world, report = lobs.import_world(
        p,
        motion="auto",
        profile=dict(extends="robotics", on_unsupported_joint="fix"),
        out=tmp_path / "ok.lobs",
    )
    assert world.spec.actors[0].joints[0].type == "fixed"
    assert any(d.field == "joint" and d.review for d in report.decisions.items)


@pytest.mark.parametrize("be", BACKENDS)
def test_urdf_arm_simulates_within_limits_and_conserves_energy(urdf, tmp_path, be):
    path, kw = urdf
    world, _ = lobs.import_world(
        path, motion="auto", profile="robotics", out=tmp_path / "a.lobs", **kw
    )
    arm = next(a for a in world.spec.actors if len(a.bodies) > 1)
    h = world.actor(arm.name)
    world.measure("q", props.joint_position(h.joint("shoulder")))
    world.measure("s", props.joint_position(h.joint("slide")))
    world.measure("energy", props.mechanical_energy(h))
    arm_h = world.actor(arm.name)
    arm_h.set_initial(q={"shoulder": 1.1})  # start near the upper limit; gravity swings it down
    report = world.validate(be)
    if not report.ok:
        pytest.skip(report.format())
    sim = world.simulate(be)
    sim.run(seconds=2.0)
    q, s = sim.history["q"], sim.history["s"]
    assert (
        q.max() <= 1.2 + 0.06
        and q.min() >= -1.0 - 0.06
        and s.max() <= 0.2 + 0.02
        and s.min() >= -0.1 - 0.02
    )
    # the arm points up from its pivot (an inverted pendulum): it falls onto the limit on the side it leans
    assert q[-1] == pytest.approx(1.2, abs=0.05)
    res = lobs.check_world(world, be, seconds=1.0)
    assert res.passed, res.summary()


def test_physically_impossible_source_inertia_is_repaired_and_flagged(imported):
    world, report = imported
    d = next(d for d in report.decisions.items if d.target == "upper" and d.field == "inertia")
    assert d.review and d.confidence == "low" and "A + B >= C" in d.rationale
    assert world.validate().ok  # the repaired tensor is valid ...
    # ... and validate() rejects such tensors when they come from anywhere else
    w = lobs.World()
    a = w.add_actor("bad")
    a.add_body(
        "b",
        inertial=lobs.schema.InertialSpec(
            mode="explicit", mass=1.0, inertia=((0.1, 0, 0), (0, 0.2, 0), (0, 0, 0.05))
        ),
        geoms=[lobs.Box(size=(0.1, 0.1, 0.1))],
    )
    assert any("inertia" in i.message for i in w.validate().errors)
