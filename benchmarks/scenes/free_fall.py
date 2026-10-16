"""Free fall: position vs analytic, mechanical energy conserved."""

import numpy as np

import lobs
from lobs import props

NAME = "free_fall"
SECONDS = 0.8
Z0, G = 10.0, 9.81
TOL = {"mujoco": {"pos_k": 2.0, "energy": 0.01}, "chrono": {"pos_k": 2.0, "energy": 0.01}}


def build(dt=0.001):
    w = lobs.World(NAME, dt=dt)
    b = w.add_box("b", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, Z0))
    w.measure("z", props.height(b))
    w.measure("me", props.mechanical_energy(b))
    return w


def check(traj, backend, dt=0.001):
    tol = TOL[backend]
    t = traj.t
    z = traj.measures["z"]
    err = np.abs(z - (Z0 - 0.5 * G * t**2))
    assert np.all(err <= tol["pos_k"] * G * dt * t + 1e-9), err.max()
    me = traj.measures["me"]
    assert np.abs(me - me[0]).max() / abs(me[0]) < tol["energy"]
