"""Actuated body tracking a reference trajectory with a position actuator."""

import numpy as np

import lobs
from lobs import props

NAME = "actuated_tracking"
SECONDS = 4.0
TOL = {"mujoco": {"rms": 0.03}, "chrono": {"rms": 0.05}}
AMP, FREQ = 0.5, 0.5


def ref(t):
    return AMP * np.sin(2 * np.pi * FREQ * t)


def controller(sim):
    return {"motor": ref(sim.time)}


def build(dt=0.002):
    w = lobs.World(NAME, dt=dt, gravity=(0, 0, 0))
    p = w.add_actor("arm", root="fixed", pos=(0, 0, 1))
    base = p.add_body("base", geoms=[lobs.Box(size=(0.05, 0.05, 0.05))], mass=1.0)
    base.add_child(
        "link",
        pos=(0, 0, -0.25),
        joint=lobs.Hinge("j", axis=(0, 1, 0), pos=(0, 0, 0.25)),
        geoms=[lobs.Capsule(radius=0.02, length=0.5)],
        mass=0.3,
        collision_group=7,
    )
    p.add_actuator("motor", "j", kind="position", kp=100.0, kd=1.5)
    w.spec.contact.disabled_group_pairs.append((7, 7))
    w.measure("q", props.joint_position(p.joint("j")))
    return w


def check(traj, backend, dt=0.002):
    t, q = traj.t, traj.measures["q"]
    mask = t > 1.0  # skip the start-up transient
    rms = np.sqrt(np.mean((q[mask] - ref(t[mask])) ** 2))
    assert rms < TOL[backend]["rms"], rms
