"""Jointed pendulum: period vs the physical-pendulum formula, energy drift, joint-state sanity."""

import numpy as np

import lobs
from lobs import props

NAME = "pendulum"
SECONDS = 10.0
THETA0 = 0.2
TOL = {
    "mujoco": {"period": 0.01, "energy": 0.02},
    "chrono": {"period": 0.02, "energy": 0.04},
}


def build(dt=0.002):
    w = lobs.World(NAME, dt=dt)
    p = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = p.add_body("base", geoms=[lobs.Box(size=(0.05, 0.05, 0.05))], mass=1.0)
    base.add_child(
        "rod",
        pos=(0, 0, -0.5),
        joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5)),
        geoms=[lobs.Capsule(radius=0.02, length=1.0)],
        mass=0.5,
        collision_group=7,
    )
    p.set_initial(q={"hinge": THETA0})
    w.spec.contact.disabled_group_pairs.append((7, 7))
    w.measure("q", props.joint_position(p.joint("hinge")))
    w.measure("me", props.mechanical_energy(p.body("rod")))
    return w


def _expected_period(reg):
    rod = next(b for b in reg.bodies if b.name == "rod")
    d = 0.5  # pivot to COM
    inertia_pivot = rod.inertia[1, 1] + rod.mass * d * d
    t0 = 2 * np.pi * np.sqrt(inertia_pivot / (rod.mass * 9.81 * d))
    return t0 * (1 + THETA0**2 / 16)


def check(traj, backend, dt=0.002):
    tol = TOL[backend]
    from lobs.registry import build_registry

    expected = _expected_period(build_registry(traj.spec))
    q, t = traj.measures["q"], traj.t
    idx = np.where((q[:-1] > 0) & (q[1:] <= 0))[0]  # downward zero crossings
    assert len(idx) >= 4
    cross = t[idx] + (t[idx + 1] - t[idx]) * q[idx] / (q[idx] - q[idx + 1])
    period = np.mean(np.diff(cross))
    assert abs(period - expected) / expected < tol["period"], (period, expected)
    me = traj.measures["me"]
    assert (me.max() - me.min()) / abs(me.max()) < tol["energy"]
