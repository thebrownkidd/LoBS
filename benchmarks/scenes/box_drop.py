"""Rigid body drop and contact response: settles at the right height, supports its own weight."""

import numpy as np

import lobs
from lobs import props

NAME = "box_drop"
SECONDS = 2.0
TOL = {
    "mujoco": {"height": 2e-3, "speed": 1e-2, "force": 0.05},
    "chrono": {"height": 3e-3, "speed": 2e-2, "force": 0.10},
}


def build(dt=0.002):
    w = lobs.World(NAME, dt=dt, substeps=2)
    w.add_plane("ground")
    b = w.add_box("box", (0.2, 0.2, 0.2), mass=2.0, pos=(0, 0, 0.6))
    w.measure("h", props.height(b))
    w.measure("speed", props.speed(b))
    w.measure("force", props.contact_force(b))
    return w


def check(traj, backend, dt=0.002):
    tol = TOL[backend]
    m = traj.measures
    assert abs(m["h"][-1] - 0.1) < tol["height"], m["h"][-1]
    assert m["speed"][-1] < tol["speed"]
    # average over the last 0.2 s (contact force is noisy tick to tick)
    f = np.mean(m["force"][-20:])
    assert abs(f - 2.0 * 9.81) < tol["force"] * 2.0 * 9.81, f
