import numpy as np
import pytest

import lobs
from lobs import props

gymnasium = pytest.importorskip("gymnasium")


def drop_world():
    w = lobs.World("drop", dt=0.002, substeps=5)
    w.add_box("floor", (10, 10, 0.2), fixed=True, pos=(0, 0, -0.1))
    w.add_sphere("ball", 0.1, mass=0.5, pos=(0, 0, 2.0), restitution=0.5)
    w.measure("z", props.position("ball").z)
    w.measure("vz", props.velocity("ball").z)
    return w


def make(skip=1, **kw):
    return lobs.GymEnv(drop_world(), "mujoco", observe=["z", "vz"], reward="z", max_seconds=3.0, frame_skip=skip, **kw)


def test_frame_skip_is_k_ticks_with_one_observation():
    a, b = make(1), make(5)
    a.reset(seed=0)
    b.reset(seed=0)
    for _ in range(5):
        oa, *_ = a.step(np.zeros(0))
    ob, *_ = b.step(np.zeros(0))
    assert np.array_equal(oa, ob)
    assert b.sim.tick == 5 and a.sim.tick == 5
    with pytest.raises(lobs.errors.LobsError):
        make(0)


def test_terminate_each_tick_stops_the_decision_early():
    env = lobs.GymEnv(drop_world(), "mujoco", observe=["z"], terminated=lambda s: s["z"] < 1.5, max_seconds=3.0,
                      frame_skip=200, terminate_each_tick=True)  # fmt: skip
    env.reset(seed=0)
    _, _, term, _, _ = env.step(np.zeros(0))
    assert term and env.sim.tick < 200  # stopped when the ball crossed 1.5 m, not after all 200 ticks
    late = lobs.GymEnv(drop_world(), "mujoco", observe=["z"], terminated=lambda s: s["z"] < 1.5, max_seconds=3.0, frame_skip=200)
    late.reset(seed=0)
    late.step(np.zeros(0))
    assert late.sim.tick == 200


@pytest.mark.parametrize("asynchronous", [False, True])
def test_make_vec_shapes_and_per_env_determinism(asynchronous):
    venv = lobs.make_vec(drop_world(), 3, observe=["z", "vz"], reward="z", max_seconds=1.0, frame_skip=5, asynchronous=asynchronous)
    try:
        obs, _ = venv.reset(seed=10)
        assert obs.shape == (3, 2)
        for _ in range(4):
            obs, rew, term, trunc, _ = venv.step(np.zeros((3, 0)))
        assert obs.shape == (3, 2) and rew.shape == (3,)
        first = obs.copy()
        venv.reset(seed=10)
        for _ in range(4):
            obs, *_ = venv.step(np.zeros((3, 0)))
        assert np.array_equal(first, obs)  # same seeds, same trajectories
    finally:
        venv.close()
