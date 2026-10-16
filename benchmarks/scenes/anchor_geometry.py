"""Two-anchor distance and relative-angle measurements on a static configuration (exact)."""

import numpy as np

import lobs
from lobs import rel

NAME = "anchor_geometry"
SECONDS = 0.02
TOL = {"mujoco": {"abs": 1e-9}, "chrono": {"abs": 1e-9}}
A, B = np.array([1.0, 2.0, 3.0]), np.array([-2.0, 0.5, 4.5])


def build(dt=0.002):
    w = lobs.World(NAME, dt=dt)
    w.add_plane("ground")
    a, b = w.anchor("a", pos=tuple(A)), w.anchor("b", pos=tuple(B))
    p = rel.between(a, b)
    w.measure("dist", p.distance())
    w.measure("polar", p.angle("polar", axis="z", frame="world"))
    w.measure("azim", p.angle("azimuth", axis="z", frame="world"))
    w.measure("elev", p.angle("elevation", axis="z", frame="world"))
    return w


def check(traj, backend, dt=0.002):
    tol = TOL[backend]["abs"]
    d = B - A
    r = np.linalg.norm(d)
    m = traj.measures
    assert np.allclose(m["dist"], r, atol=tol)
    assert np.allclose(m["polar"], np.arccos(d[2] / r), atol=tol)
    assert np.allclose(m["elev"], np.pi / 2 - np.arccos(d[2] / r), atol=tol)
    assert np.allclose(m["azim"], np.arctan2(d[1], d[0]), atol=tol)
