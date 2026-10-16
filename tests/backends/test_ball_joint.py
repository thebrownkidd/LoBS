"""Ball joints (cross-engine): kinematics convention, free swing, damping, measures, replay."""

import numpy as np
import pytest

import lobs
from lobs import math3d as m3
from lobs import props
from lobs.kinematics import fk_actor
from lobs.registry import resolve_initial

BACKENDS = lobs.backends.available()
pytestmark = pytest.mark.parametrize("be", BACKENDS)


def ball_world(gravity=(0, 0, 0), damping=0.0):
    w = lobs.World(dt=0.001, gravity=gravity)
    a = w.add_actor("arm", root="fixed", pos=(0, 0, 2))
    base = a.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0, collision_group=4)
    j = lobs.schema.JointSpec(name="shoulder", type="ball", pos=(0, 0, 0.5), damping=damping)
    rod = base.add_child(
        "rod",
        pos=(0, 0, -0.5),
        joint=j,
        geoms=[lobs.Capsule(radius=0.02, length=1.0)],
        mass=0.5,
        collision_group=4,
    )
    w.spec.contact.disabled_group_pairs.append((4, 4))
    return w, a, rod


def test_ball_kinematics_match_lobs_fk_at_reset_and_while_moving(be):
    w, a, rod = ball_world()
    a.set_initial(q={"shoulder": (0.3, -0.5, 0.2)}, qd={"shoulder": (0.4, 0.1, -0.6)})
    sim = w.simulate(be)
    reg = sim.reg
    ri = resolve_initial(reg, None)
    fk = fk_actor(reg, 0, ri.actor_pose[0], ri.q, ri.qd, np.zeros(3), np.zeros(3))
    f = sim.state()
    p, q, v, om = fk[1]
    k = reg.bodies[1].dyn
    assert np.allclose(f.body_pos[k], p, atol=1e-9) and m3.quat_same_rotation(
        f.body_quat[k], q, 1e-8
    )
    assert np.allclose(f.body_linvel[k], v, atol=1e-9) and np.allclose(
        f.body_angvel[k], om, atol=1e-9
    )
    assert np.allclose(f.joint_q, (0.3, -0.5, 0.2), atol=1e-9) and np.allclose(
        f.joint_qd, (0.4, 0.1, -0.6), atol=1e-9
    )
    tol = 1e-8 if be == "mujoco" else 1e-3
    for _ in range(
        4
    ):  # still consistent while swinging freely (no gravity: pure rotation about the joint point)
        sim.run(ticks=100)
        f = sim.state()
        fk = fk_actor(
            reg, 0, ((0, 0, 2.0), (1, 0, 0, 0)), f.joint_q, f.joint_qd, np.zeros(3), np.zeros(3)
        )
        p, q, v, om = fk[1]
        assert np.allclose(f.body_pos[k], p, atol=tol) and m3.quat_same_rotation(
            f.body_quat[k], q, 10 * tol
        )
        assert np.allclose(f.body_angvel[k], om, atol=10 * tol) and np.allclose(
            f.body_linvel[k], v, atol=10 * tol
        )


def test_ball_joint_pendulum_swings_in_a_plane_like_a_hinge(be):
    w, a, rod = ball_world(gravity=(0, 0, -9.81))
    a.set_initial(q={"shoulder": (0.0, 0.3, 0.0)})  # rotation about +y by 0.3 rad
    w.measure("q", props.joint_position(a.joint("shoulder")))
    w.measure("qd", props.joint_velocity(a.joint("shoulder")))
    w.measure("tip_h", props.height(rod))
    w.measure("energy", props.mechanical_energy(rod))
    sim = w.simulate(be)
    sim.run(seconds=2.0)
    q = sim.history["q"]
    assert q.shape[1] == 3
    assert np.abs(q[:, 0]).max() < 0.02 and np.abs(q[:, 2]).max() < 0.02  # stays in the y-plane
    assert q[:, 1].max() > 0.28 and q[:, 1].min() < -0.28  # swings through the bottom
    e = sim.history["energy"]
    assert (e.max() - e.min()) / abs(e.max()) < (0.01 if be == "mujoco" else 0.05)


def test_ball_joint_damping_dissipates_energy(be):
    w, a, rod = ball_world(damping=0.5)
    a.set_initial(qd={"shoulder": (0.0, 3.0, 0.0)})
    w.measure("spin", props.spin(rod))
    sim = w.simulate(be)
    s0 = sim["spin"]
    sim.run(seconds=1.0)
    assert sim["spin"] < 0.35 * s0  # I ~ 0.04: exp(-0.5/0.04 * 1) is tiny; allow solver slack


def test_ball_measures_rates_and_validation(be):
    w, a, rod = ball_world()
    a.set_initial(qd={"shoulder": (0.0, 1.0, 0.0)})
    j = a.joint("shoulder")
    q = w.measure("q", props.joint_position(j))
    w.measure("q_rate", q.rate())  # exact rate of the rotation vector along the ball's own motion
    sim = w.simulate(be)
    sim.run(seconds=0.5)
    assert np.allclose(sim["q_rate"], (0.0, 1.0, 0.0), atol=2e-3)  # rotation about y at 1 rad/s
    bad = ball_world()[0]
    bad.measure("eff", props.joint_effort(bad.actor("arm").joint("shoulder")))
    assert any("ball" in i.message for i in bad.validate().errors)
    ranged = ball_world()[0]
    ranged.spec.actors[0].joints[0].range = (-1.0, 1.0)
    assert not ranged.validate(be).ok  # ball limits unsupported
    w2, _, _ = ball_world()
    with pytest.raises(lobs.errors.LobsError, match="3-vector"):
        resolve_initial(w2.registry(), {"arm": lobs.Initial(q={"shoulder": 0.5})})
