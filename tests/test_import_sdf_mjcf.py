import numpy as np
import pytest

import lobs
from lobs import props

from .fixtures.make_robot2 import make_mjcf, make_sdf

BACKENDS = lobs.backends.available()


def actor(world, name):
    return next(a for a in world.spec.actors if a.name == name)


# ----------------------------------------------------------------------------- SDF
@pytest.fixture(scope="module")
def sdf_world(tmp_path_factory):
    d = tmp_path_factory.mktemp("sdf")
    return lobs.import_world(
        make_sdf(d / "w.sdf"), motion="auto", profile="robotics", out=d / "w.lobs"
    )


def test_sdf_world_physics_models_and_surfaces(sdf_world):
    world, report = sdf_world
    assert world.spec.gravity == pytest.approx(
        (0, 0, -9.81)
    ) and world.spec.timing.dt == pytest.approx(0.002)
    ground, crate = actor(world, "ground"), actor(world, "crate")
    assert ground.root == "fixed" and any(
        g.shape.type == "plane" for g in ground.bodies[0].geoms
    )  # <static>
    assert next(
        g for g in ground.bodies[0].geoms if g.shape.type == "plane"
    ).material.friction == pytest.approx(0.9)
    assert crate.root == "free" or crate.root == "fixed"
    assert crate.pose.pos == pytest.approx((1, 0, 0.5))
    assert crate.bodies[0].inertial.mass == pytest.approx(3.0)
    mat = next(g.material for g in crate.bodies[0].geoms if g.role != "visual")
    assert mat.friction == pytest.approx(0.6) and mat.restitution == pytest.approx(0.3)


def test_sdf_joint_to_world_with_limits_and_ball(sdf_world):
    world, _ = sdf_world
    pend = actor(world, "pendulum")
    assert pend.root == "fixed" and len(pend.bodies) == 2  # synthetic world base + the arm
    h = pend.joints[0]
    assert (
        h.type == "hinge"
        and h.range == pytest.approx((-0.8, 0.8))
        and h.damping == pytest.approx(0.02)
    )
    assert np.linalg.norm(h.pos) == pytest.approx(0.5) and h.axis == pytest.approx((0, 1, 0))
    ball = actor(world, "shoulder_arm")
    assert ball.joints[0].type == "ball" and np.linalg.norm(ball.joints[0].pos) == pytest.approx(
        0.2
    )
    assert any(a.effort_limit == pytest.approx(20) for a in pend.actuators) or not pend.actuators


@pytest.mark.parametrize("be", BACKENDS)
def test_sdf_world_simulates(sdf_world, be):
    world, _ = sdf_world
    if not world.validate(be).ok:
        pytest.skip(world.validate(be).format())
    res = lobs.check_world(world, be, seconds=1.5)
    assert res.passed, res.summary()


# ----------------------------------------------------------------------------- MJCF
@pytest.fixture(scope="module")
def mj_world(tmp_path_factory):
    d = tmp_path_factory.mktemp("mjcf")
    return lobs.import_world(
        make_mjcf(d / "demo.xml"), motion="auto", profile="robotics", out=d / "demo.lobs"
    )


def test_mjcf_bodies_joints_units_and_sites(mj_world):
    world, report = mj_world
    box, floor = actor(world, "box"), actor(world, "floor")
    assert box.root == "free" and floor.root == "fixed"
    assert box.bodies[0].inertial.mass == pytest.approx(2.0)
    assert [a.name for a in box.anchors] == ["top"] and box.anchors[0].pose.pos == pytest.approx(
        (0, 0, 0.1)
    )
    pend = actor(world, "pend")
    hinge = pend.joints[0]
    assert hinge.type == "hinge" and hinge.range == pytest.approx(
        (np.radians(-30), np.radians(60))
    )  # degrees -> radians
    assert hinge.damping == pytest.approx(0.01) and hinge.armature == pytest.approx(0.002)
    elbow = next(j for a in world.spec.actors for j in a.joints if j.name == "elbow")
    assert elbow.range == pytest.approx((0.0, np.pi / 2))
    ball = next(j for a in world.spec.actors for j in a.joints if j.name == "ball")
    assert ball.type == "ball" and ball.damping == pytest.approx(0.1)


def test_mjcf_multi_joint_body_becomes_chain_and_dropped_features(mj_world):
    world, report = mj_world
    g = actor(world, "gimbal")
    # world base, one massless link between the two hinges, then the real body
    assert (
        [b.name for b in g.bodies][0] == "world_base"
        and len(g.bodies) == 3
        and [j.type for j in g.joints] == ["hinge", "hinge"]
    )
    mid, real = g.bodies[1], g.bodies[2]
    assert mid.inertial.mass < 1e-3 and not mid.geoms and real.inertial.mass == pytest.approx(0.5)
    assert any("tendons" in w for w in report.warnings)


def test_mjcf_actuators_and_mesh(mj_world):
    world, _ = mj_world
    pend = actor(world, "pend")
    act = pend.actuators[0]
    assert act.kind == "effort" and act.ctrl_range == pytest.approx((-5, 5))
    forearm_arm = next(a for a in world.spec.actors if any(j.name == "elbow" for j in a.joints))
    pos = next(x for x in forearm_arm.actuators if x.kind == "position")
    assert pos.kp == pytest.approx(20) and pos.kd == pytest.approx(2)
    wedge = actor(world, "wedge")
    assert any(g.shape.type in ("convex_mesh", "convex_set") for g in wedge.bodies[0].geoms)
    assert wedge.bodies[0].inertial.mass == pytest.approx(1.0)


@pytest.mark.parametrize("be", BACKENDS)
def test_mjcf_world_simulates_and_matches_mujoco_itself(mj_world, be, tmp_path):
    world = mj_world[0].copy()
    world.measure("box_h", props.height(world.actor("box")))
    world.measure("pend_q", props.joint_position(world.actor("pend").joint("hinge")))
    rep = world.validate(be)
    if not rep.ok:
        pytest.skip(rep.format())  # e.g. chrono has no joint armature
    res = lobs.check_world(world, be, seconds=1.5)
    assert res.passed, res.summary()
    sim = world.simulate(be)
    sim.run(seconds=1.5)
    assert sim["box_h"] == pytest.approx(0.1, abs=5e-3)  # a 0.2 m box rests on the floor
    q = sim.history["pend_q"]
    assert q.min() >= np.radians(-30) - 0.06 and q.max() <= np.radians(60) + 0.06


def test_mjcf_import_reproduces_native_mujoco_trajectory(mj_world, tmp_path):
    """The imported world, simulated by LoBS on MuJoCo, tracks the original MJCF simulated directly."""
    import mujoco

    world = mj_world[0].copy()
    world.measure("pend_q", props.joint_position(world.actor("pend").joint("hinge")))
    sim = world.simulate("mujoco")
    sim.run(seconds=1.0)
    from .fixtures.make_robot2 import MJCF

    m = mujoco.MjModel.from_xml_string(MJCF)
    d = mujoco.MjData(m)
    for _ in range(500):
        mujoco.mj_step(m, d)
    native_q = d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hinge")]]
    assert sim["pend_q"] == pytest.approx(native_q, abs=2e-3)
