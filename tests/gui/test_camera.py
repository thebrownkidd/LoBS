import numpy as np
import pytest

import lobs
from lobs.gui.core.camera import OrbitCamera, pick_sphere


def test_project_and_ray_are_inverse():
    cam = OrbitCamera(lookat=np.array([1.0, 2.0, 0.5]), distance=6.0, azimuth=40.0, elevation=-30.0)
    p = np.array([1.7, 2.4, 1.1])
    pix, z = cam.project(p, 640, 480)
    assert z[0] > 0
    o, d = cam.ray(float(pix[0, 0]), float(pix[0, 1]), 640, 480)
    # the ray through the projected pixel passes through the point
    t = np.dot(p - o, d)
    assert np.linalg.norm(o + t * d - p) < 1e-9
    # the look-at point projects to the image centre
    centre, _ = cam.project(cam.lookat, 640, 480)
    assert centre[0] == pytest.approx((320.0, 240.0))


def test_projection_matches_the_real_renderer():
    w = lobs.World("cam", gravity=(0, 0, 0))
    w.add_sphere("ref", 0.05, mass=1.0, pos=(0, 0, 0), fixed=True)
    marker = w.add_sphere("marker", 0.15, mass=1.0, pos=(0.8, -0.5, 0.6), fixed=True)  # noqa: F841
    sim = w.simulate("mujoco")
    cam = {
        "lookat": np.array([0.0, 0.0, 0.0]),
        "distance": 4.0,
        "azimuth": 120.0,
        "elevation": -20.0,
    }
    img = sim.render(480, 360, camera=cam).astype(int)
    base = sim.render(480, 360, camera=cam)
    # find the marker by differencing against a world without it
    w2 = lobs.World("cam2", gravity=(0, 0, 0))
    w2.add_sphere("ref", 0.05, mass=1.0, pos=(0, 0, 0), fixed=True)
    other = w2.simulate("mujoco").render(480, 360, camera=cam).astype(int)
    diff = np.abs(img - other).sum(axis=2) > 30
    assert diff.sum() > 50
    oc = OrbitCamera(**cam)
    pix, _ = oc.project([0.8, -0.5, 0.6], 480, 360)
    yy, xx = np.mgrid[0:360, 0:480]
    near = diff & (
        (xx - pix[0, 0]) ** 2 + (yy - pix[0, 1]) ** 2 < 40**2
    )  # ignore shadow / floor changes
    assert near.sum() > 100
    assert abs(xx[near].mean() - pix[0, 0]) < 4 and abs(yy[near].mean() - pix[0, 1]) < 4
    assert base.shape == (360, 480, 3)


def test_pick_sphere_nearest_and_miss():
    o, d = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert pick_sphere(o, d, [(5, 0, 0), (2, 0, 0)], [0.5, 0.5]) == 1
    assert pick_sphere(o, d, [(5, 3, 0)], [0.5]) is None


def test_orbit_pan_zoom_stay_sane():
    cam = OrbitCamera()
    cam.orbit(1000, 1000)
    assert -89.0 <= cam.elevation <= 89.0
    cam.zoom(5)
    assert cam.distance < 5.0
    before = cam.lookat.copy()
    cam.pan(10, 0, 480)
    assert not np.allclose(before, cam.lookat)
