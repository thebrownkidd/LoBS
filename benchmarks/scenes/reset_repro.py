"""Reset reproducibility and trajectory export: two runs from reset are identical."""

import numpy as np

import lobs
from lobs import props

NAME = "reset_repro"
SECONDS = 1.0
TOL = {"mujoco": {"rtol": 0.0}, "chrono": {"rtol": 1e-9}}


def build(dt=0.002):
    w = lobs.World(NAME, dt=dt, substeps=2)
    w.add_plane("ground")
    b = w.add_box("box", (0.2, 0.3, 0.1), mass=1.0, pos=(0, 0, 0.5), angvel=(1.0, 2.0, 3.0), linvel=(0.2, 0, 0))
    w.measure("pos", props.position(b))
    return w


def run_twice(sim):
    sim.run(seconds=SECONDS)
    first = sim.history["pos"].copy()
    sim.reset()
    sim.run(seconds=SECONDS)
    return first, sim.history["pos"].copy()


def check_pair(first, second, backend):
    rtol = TOL[backend]["rtol"]
    if rtol == 0.0:
        assert np.array_equal(first, second)
    else:
        assert np.allclose(first, second, rtol=rtol, atol=1e-12)
