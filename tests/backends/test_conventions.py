"""Cross-engine semantic contract (plan section 4). Parametrized over every available backend."""

import numpy as np
import pytest

import lobs
from lobs import backends, props
from lobs import math3d as m3
from lobs.kinematics import fk_actor
from lobs.registry import resolve_initial
from lobs.schema import GeomSpec

BACKENDS = backends.available() or ["mujoco"]
pytestmark = pytest.mark.parametrize("be", BACKENDS)


def qaxis(axis, ang):
    return m3.quat_from_axis_angle(axis, ang)


def test_box_rest_height_force_and_contact_orientation(be):
    w = lobs.World(dt=0.002, substeps=2)
    w.add_plane("ground")
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=2.0, pos=(0, 0, 0.5))
    w.measure("h", props.height(box))
    w.measure("f", props.contact_force(box))
    w.measure("touch", props.in_contact(box))
    w.measure("acc", props.acceleration(box))
    sim = w.simulate(be)
    sim.run(seconds=1.5)
    assert sim["h"] == pytest.approx(0.1, abs=2e-3)  # size is FULL extents
    assert sim["f"] == pytest.approx(2.0 * 9.81, rel=0.02)
    assert sim["touch"] is True
    assert np.allclose(sim["acc"], 0, atol=0.05)  # coordinate acceleration at rest
    c = sim.state().contacts
    assert len(c) > 0
    assert np.all(c.body_a < c.body_b)
    assert np.allclose(c.normal, (0, 0, 1), atol=1e-6)  # from A (ground) to B (box)
    assert np.all(c.depth >= 0) and np.all(c.normal_force >= 0)
    assert np.allclose(c.force_on_b.sum(axis=0), (0, 0, 2.0 * 9.81), rtol=0.02, atol=0.05)


def test_free_fall_position_and_acceleration(be):
    w = lobs.World(dt=0.001)
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 10))
    w.measure("acc", props.acceleration(box))
    sim = w.simulate(be)
    sim.run(seconds=0.5)
    assert sim.state().body_pos[0][2] == pytest.approx(10 - 0.5 * 9.81 * 0.25, abs=6e-3)
    assert np.allclose(sim["acc"], (0, 0, -9.81), atol=1e-6)


def test_offset_com_reference_velocity_semantics(be):
    w2 = lobs.World(dt=0.0005, gravity=(0, 0, 0))
    act = w2.add_actor("a", root="free", pos=(0, 0, 1))
    shifted = GeomSpec(shape=lobs.Box(size=(0.4, 0.2, 0.2)), pose=lobs.Pose(pos=(0.3, 0, 0)))
    act.add_body("b", geoms=[shifted], mass=1.0)
    act.set_initial(linvel=(1, 0, 0), angvel=(0, 0, 2))
    w2.measure("v_com", props.velocity(act))
    w2.measure("v_ref", props.velocity(act.origin))
    sim = w2.simulate(be)
    f0 = sim.state()
    assert np.allclose(f0.body_linvel[0], (1, 0, 0), atol=1e-9)  # reference origin velocity
    assert np.allclose(f0.body_angvel[0], (0, 0, 2), atol=1e-9)
    assert np.allclose(sim["v_com"], (1, 0.6, 0), atol=1e-9)  # v_ref + w x (R c)
    assert np.allclose(sim["v_ref"], (1, 0, 0), atol=1e-9)
    sim.run(ticks=400)
    traj = sim.trajectory()
    pos, vel = traj.arrays["body_pos"][:, 0], traj.arrays["body_linvel"][:, 0]
    fd = (pos[2:] - pos[:-2]) / (2 * 0.0005)  # central difference of the REFERENCE position
    assert np.allclose(vel[1:-1], fd, atol=2e-3)


def test_free_root_angular_velocity_round_trip(be):
    w = lobs.World(gravity=(0, 0, 0))
    b = w.add_box("b", (0.3, 0.2, 0.1), mass=1.0, pos=(0, 0, 1), quat=tuple(qaxis([1, 2, 3], 0.9)))
    b.set_initial(angvel=(1.0, 2.0, 3.0), linvel=(0.5, -0.5, 0.25))
    sim = w.simulate(be)
    f = sim.state()
    assert np.allclose(f.body_angvel[0], (1, 2, 3), atol=1e-9)  # world frame in AND out
    assert np.allclose(f.body_linvel[0], (0.5, -0.5, 0.25), atol=1e-9)


def arm_world():
    w = lobs.World(dt=0.001, gravity=(0, 0, 0))
    w.spec.contact.disabled_group_pairs.append((5, 5))
    a = w.add_actor("arm", root="free", pos=(0.3, -0.2, 1.0), quat=tuple(qaxis([1, 2, 3], 0.7)))
    base = a.add_body("base", geoms=[lobs.Box(size=(0.2, 0.2, 0.2))], mass=1.0, collision_group=5)
    l1 = base.add_child("l1", pos=(0.3, 0, 0.1), quat=tuple(qaxis([0, 1, 0], 0.4)),
                        joint=lobs.Hinge("h1", axis=(0, 1, 0), pos=(0, 0, 0.0)),
                        geoms=[lobs.Capsule(radius=0.03, length=0.4)], mass=1.0, collision_group=5)  # fmt: skip
    l2 = l1.add_child("l2", pos=(0, 0.1, 0.4), joint=lobs.Slide("s2", axis=(0, 0, 1)),
                      geoms=[lobs.Box(size=(0.1, 0.1, 0.3))], mass=1.0, collision_group=5)  # fmt: skip
    l2.add_child("l3", pos=(0.05, 0, 0.3), joint=lobs.Hinge("h3", axis=(1, 1, 0), pos=(0.1, 0, 0.1)),
                 geoms=[lobs.Cylinder(radius=0.05, length=0.2)], mass=1.0, collision_group=5)  # fmt: skip
    a.set_initial(linvel=(0.1, 0.2, -0.1), angvel=(0.3, -0.2, 0.5),
                  q={"h1": 0.5, "s2": 0.1, "h3": -0.8}, qd={"h1": 0.7, "s2": -0.2, "h3": 1.1})  # fmt: skip
    return w


def assert_matches_fk(sim, atol=1e-8):
    reg, f = sim.reg, sim.state()
    root = reg.bodies[0]
    fk = fk_actor(reg, 0, (f.body_pos[0], f.body_quat[0]), f.joint_q, f.joint_qd,
                  f.body_linvel[0], f.body_angvel[0])  # fmt: skip
    assert root.parent < 0
    for gi, (p, q, v, om) in fk.items():
        k = reg.bodies[gi].dyn
        assert np.allclose(f.body_pos[k], p, atol=atol), reg.bodies[gi].path
        assert m3.quat_same_rotation(f.body_quat[k], q, 1e-7), reg.bodies[gi].path
        assert np.allclose(f.body_linvel[k], v, atol=atol), reg.bodies[gi].path
        assert np.allclose(f.body_angvel[k], om, atol=atol), reg.bodies[gi].path


def test_joint_and_twist_conventions_match_lobs_kinematics(be):
    w = arm_world()
    sim = w.simulate(be)
    # at reset (pure kinematics) the engine state must equal LoBS forward kinematics of the SPEC
    ri = resolve_initial(sim.reg, None)
    fk = fk_actor(sim.reg, 0, ri.actor_pose[0], ri.q, ri.qd, ri.linvel[0], ri.angvel[0])
    f = sim.state()
    for gi, (p, q, v, om) in fk.items():
        k = sim.reg.bodies[gi].dyn
        assert np.allclose(f.body_pos[k], p, atol=1e-9)
        assert m3.quat_same_rotation(f.body_quat[k], q, 1e-8)
        assert np.allclose(f.body_linvel[k], v, atol=1e-9)
        assert np.allclose(f.body_angvel[k], om, atol=1e-9)
    assert np.allclose(f.joint_q, [0.5, 0.1, -0.8]) and np.allclose(f.joint_qd, [0.7, -0.2, 1.1])
    # and stay consistent while the mechanism moves (twist convention under dynamics)
    for _ in range(5):
        sim.run(ticks=60)
        assert_matches_fk(sim)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_hinge_right_hand_rule_and_unwrapped_angle(be, sign):
    w = lobs.World(dt=0.002, gravity=(0, 0, 0))
    pend = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = pend.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0)
    base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5)),
                   geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5)  # fmt: skip
    pend.set_initial(qd={"hinge": 10.0 * sign})
    w.measure("q", props.joint_position(pend.joint("hinge")))
    w.measure("tip", props.position(pend.body("rod")))
    sim = w.simulate(be)
    sim.run(ticks=50)  # 0.1 s -> q = +-1 rad
    # right-hand rotation about +y moves the down-pointing rod toward -x for positive q
    assert sim["tip"][0] == pytest.approx(-0.5 * np.sin(sign * 1.0), abs=2e-2)
    assert sim["q"] == pytest.approx(sign * 1.0, abs=1e-2)
    sim.run(ticks=500)  # 1.1 s total -> 11 rad: well past 2*pi, must NOT wrap
    assert sim["q"] == pytest.approx(sign * 11.0, abs=0.05)


def test_actuator_laws_clip_then_law_then_limit(be):
    def effort_after_step(kind, u, **kw):
        w = lobs.World(dt=0.001, gravity=(0, 0, 0))
        pend = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
        base = pend.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0)
        base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5)),
                       geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5)  # fmt: skip
        pend.add_actuator("m", "hinge", kind=kind, **kw)
        pend.set_initial(q={"hinge": 0.2}, qd={"hinge": 0.3})
        w.measure("tau", props.joint_effort(pend.joint("hinge")))
        sim = w.simulate(be)
        sim.step(action={"m": u})
        return sim["tau"]

    q0, qd0 = 0.2, 0.3
    assert effort_after_step("effort", 0.7) == pytest.approx(0.7)
    assert effort_after_step("effort", 0.7, effort_limit=0.5) == pytest.approx(0.5)
    assert effort_after_step("effort", 9.0, ctrl_range=(-1, 1)) == pytest.approx(1.0)
    assert effort_after_step("position", 0.5, kp=10, kd=1) == pytest.approx(
        10 * (0.5 - q0) - 1 * qd0
    )
    assert effort_after_step("position", 5.0, kp=10, kd=1, ctrl_range=(-1, 1)) == pytest.approx(
        10 * (1.0 - q0) - qd0
    )
    assert effort_after_step("position", 5.0, kp=10, kd=1, effort_limit=3) == pytest.approx(3.0)
    assert effort_after_step("velocity", 1.0, kd=2) == pytest.approx(2 * (1.0 - qd0))


def test_off_com_wrench_produces_expected_torque(be):
    w = lobs.World(dt=0.001, substeps=10, gravity=(0, 0, 0))
    box = w.add_box("b", (1, 1, 1), mass=1.0, pos=(0, 0, 1))
    w.measure("L", props.angular_momentum(box))
    w.measure("p", props.momentum(box))
    sim = w.simulate(be)
    sim.apply_force(box, force=(0, 10, 0), at=(0.5, 0, 1.0))  # r = (0.5, 0, 0) from COM
    sim.step()
    assert np.allclose(sim["p"], (0, 10 * 0.01, 0), rtol=0.01, atol=1e-6)
    assert np.allclose(sim["L"], (0, 0, 5 * 0.01), rtol=0.02, atol=1e-6)  # tau = r x F = (0,0,5)
    before = sim["L"].copy()
    sim.step()  # one-tick disturbance: no further torque
    assert np.allclose(sim["L"], before, atol=1e-6)


def test_disabled_collision_groups_pass_through(be):
    def final_height(disable):
        w = lobs.World(dt=0.002, substeps=2)
        w.add_plane("ground")
        w.add_box("lower", (1, 1, 0.2), mass=1.0, pos=(0, 0, 0.5), fixed=True, collision_group=2)
        w.add_box("upper", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 2.0), collision_group=1)
        if disable:
            w.spec.contact.disabled_group_pairs.append((1, 2))
        sim = w.simulate(be)
        sim.run(seconds=2.0)
        return sim.state().body_pos[0][2]

    assert final_height(False) == pytest.approx(
        0.7, abs=5e-3
    )  # rests on the lower box (top at 0.6)
    assert final_height(True) == pytest.approx(0.1, abs=5e-3)  # falls through to the ground


def test_static_bodies_are_not_in_state_arrays(be):
    w = lobs.World()
    w.add_plane("ground")
    w.add_box("wall", (1, 1, 1), fixed=True, pos=(3, 0, 0.5))
    w.add_box("a", (0.1, 0.1, 0.1), mass=1, pos=(0, 0, 1))
    w.add_box("b", (0.1, 0.1, 0.1), mass=1, pos=(1, 0, 1))
    sim = w.simulate(be)
    f = sim.state()
    assert f.body_pos.shape == (2, 3) and f.body_quat.shape == (2, 4)
    assert sim.reg.dynamic_order == [2, 3]


def test_reset_is_bitwise_reproducible_and_clears_state(be):
    w = lobs.World(dt=0.002, substeps=2)
    w.add_plane("ground")
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1), angvel=(1, 2, 3))
    w.measure("h", props.height(box))
    w.measure("disp", props.displacement(box))
    sim = w.simulate(be, seed=3)
    sim.run(seconds=1.0)
    first = sim.history["h"].copy()
    traj1 = sim.trajectory().arrays["body_pos"].copy()
    sim.reset()
    assert len(sim.history) == 1 and np.allclose(sim["disp"], 0)
    sim.run(seconds=1.0)
    assert np.array_equal(first, sim.history["h"])
    assert np.array_equal(traj1, sim.trajectory().arrays["body_pos"])
    sim.reset(initial_state={"box": lobs.Initial(pos=(0, 0, 2))})
    assert sim["h"] == pytest.approx(2.0)


def test_capabilities_and_restitution_warning(be):
    w = lobs.World()
    w.add_plane("ground", restitution=0.5)
    w.add_box("b", (0.1, 0.1, 0.1), mass=1, pos=(0, 0, 1), restitution=0.5)
    rep = w.validate(be)
    assert rep.ok
    assert any("restitution" in i.message for i in rep.warnings) or be != "mujoco"
    w2 = lobs.World()
    w2.add_actor("m", root="fixed").add_body("b", geoms=[lobs.schema.TriangleMesh(file="x.obj")])
    assert not w2.validate(be).ok  # triangle meshes: unsupported (mujoco) / not yet wired
