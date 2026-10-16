"""Benchmark scenes run on every available backend (plan section 11.7). Marked ``bench``."""

import numpy as np
import pytest

import lobs
from benchmarks.scenes import (
    actuated_tracking,
    anchor_geometry,
    box_drop,
    free_fall,
    pendulum,
    reset_repro,
)

BACKENDS = lobs.backends.available() or ["mujoco"]
pytestmark = pytest.mark.bench

SCENES = [free_fall, box_drop, pendulum, actuated_tracking, anchor_geometry]


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("scene", SCENES, ids=lambda s: s.NAME)
def test_scene(scene, be):
    sim = scene.build().simulate(be, seed=0)
    controller = getattr(scene, "controller", None)
    sim.run(seconds=scene.SECONDS, controller=controller)
    scene.check(sim.trajectory(), be)


@pytest.mark.parametrize("be", BACKENDS)
def test_reset_reproducibility(be):
    sim = reset_repro.build().simulate(be, seed=0)
    first, second = reset_repro.run_twice(sim)
    reset_repro.check_pair(first, second, be)


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("dt", [0.0005, 0.001, 0.002, 0.005])
def test_dt_sweep_is_stable(be, dt):
    for scene in (box_drop, pendulum):
        sim = scene.build(dt=dt).simulate(be)
        sim.run(seconds=2.0)
        for name in sim.history.names():
            assert np.all(np.isfinite(sim.history[name])), (scene.NAME, name, dt)
        if scene is box_drop:
            assert 0.05 < sim["h"] < 0.15, (dt, sim["h"])


@pytest.mark.parametrize("be", BACKENDS)
def test_trajectory_export_roundtrip(be, tmp_path):
    sim = box_drop.build().simulate(be)
    sim.run(seconds=0.5)
    p = sim.export(tmp_path / "t.npz")
    t = lobs.Trajectory.load(p)
    assert np.array_equal(t.measures["h"], sim.history["h"])
    assert t.meta["backend"]["engine"] == be
    assert np.allclose(t.evaluate("h"), sim.history["h"])
