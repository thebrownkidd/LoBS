import numpy as np
import pytest

from demos.lobpass import constants as C
from demos.lobpass.machines import DUNKER_STATES
from demos.lobpass.scenario import sample_scenario
from demos.lobpass.world import build_world, dyn_slots, to_reset_args


@pytest.fixture(scope="module")
def sim_slots():
    sim = build_world("train").simulate("mujoco", seed=0, record=False)
    return sim, dyn_slots(sim)


def run_dunker(sim, slots, scn, seconds=7.0):
    init, rnd = to_reset_args(scn)
    sim.reset(seed=scn.seed, initial_state=init, randomize=rnd)
    path, states = [], []
    for _ in range(int(seconds / C.TICK)):
        sim.step()
        path.append(sim.state().body_pos[slots["dunker"]].copy())
        states.append(int(sim["dunker_state"]))
    return np.array(path), np.array(states)


def test_dunker_runs_corner_then_takes_off_at_the_takeoff_point(sim_slots):
    sim, slots = sim_slots
    n_corner = n_total = 0
    for seed in range(40):
        scn = sample_scenario("S1", seed)
        path, states = run_dunker(sim, slots, scn)
        names = [DUNKER_STATES[s] for s in states]
        assert "jump" in names, (seed, scn.corner)
        k = names.index("jump")
        takeoff = np.array(scn.takeoff)
        assert np.linalg.norm(path[k][:2] - takeoff) <= 0.45, (
            seed,
            np.linalg.norm(path[k][:2] - takeoff),
        )
        if scn.corner is not None:
            n_corner += 1
            assert "run_c" in names
            assert np.min(np.linalg.norm(path[:, :2] - np.array(scn.corner), axis=1)) <= 0.9
        n_total += 1
    assert n_corner > 5


def test_dunker_jumps_to_his_leap_and_comes_down(sim_slots):
    sim, slots = sim_slots
    scn = sample_scenario("S0", 3)
    path, states = run_dunker(sim, slots, scn, seconds=9.0)
    assert abs(path[:, 2].max() - scn.dunker_L) < 0.012
    assert DUNKER_STATES[states[-1]] == "land"
    assert path[-1][2] < 0.01


def test_takeoff_time_is_plausible_and_repeatable(sim_slots):
    sim, slots = sim_slots
    for seed in range(15):
        scn = sample_scenario("S1", seed)
        a, sa = run_dunker(sim, slots, scn)
        b, sb = run_dunker(sim, slots, scn)
        ka, kb = int(np.argmax(sa == 3)), int(np.argmax(sb == 3))
        assert ka == kb and np.array_equal(a, b)
        pts = (
            [np.array(scn.dunker_start)]
            + ([np.array(scn.corner)] if scn.corner else [])
            + [np.array(scn.takeoff)]
        )
        length = sum(np.linalg.norm(q - p) for p, q in zip(pts[:-1], pts[1:], strict=True))
        t = ka * C.TICK
        assert (
            length / scn.vmax + scn.start_delay
            <= t
            <= length / min(scn.v_gather, scn.v_corner) + scn.start_delay + 1.5
        )


def test_native_core_survives_machines():
    sim = build_world("train").simulate("mujoco", record=False)
    assert sim.info["native_core"] is True
