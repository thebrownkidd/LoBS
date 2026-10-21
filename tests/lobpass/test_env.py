import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import lobs
from demos.lobpass import ballistics as B
from demos.lobpass import constants as C
from demos.lobpass import obs as O
from demos.lobpass import oracle
from demos.lobpass.env import LobPassEnv
from demos.lobpass.scenario import ROLE_STATIC, DefenderSpec, sample_scenario

NO_THROW = np.array([-1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def env():
    return LobPassEnv({"S0": 1.0})


def play(env, scn, actions):
    env.reset(options={"scenario": scn})
    out = None
    for a in actions:
        out = env.step(a)
        if out[2]:
            break
    return out


def test_observation_layout_and_range(env):
    assert len(O.NAMES) == C.OBS_DIM == 109 and len(set(O.NAMES)) == 109
    obs, _ = env.reset(options={"scenario": sample_scenario("S3c", 5)})
    assert (
        obs.shape == (109,)
        and obs.dtype == np.float32
        and np.isfinite(obs).all()
        and np.abs(obs).max() <= 5.0
    )
    assert obs[O.NAMES.index("n_def")] == pytest.approx(env.scn.n_def / 5)
    assert obs[O.NAMES.index("def0_present")] == 1.0 and obs[O.NAMES.index("def4_present")] in (
        0.0,
        1.0,
    )
    for k in range(env.scn.n_def, 5):
        assert not obs[29 + k * 16 : 29 + (k + 1) * 16].any()  # empty slots are all zero


def test_gymnasium_checker():
    e = LobPassEnv({"S0": 1.0})
    check_env(e, skip_render_check=True)
    e.close()


def test_no_throw_is_no_pass(env):
    scn = sample_scenario("S0", 1)
    _, r, term, _, info = play(env, scn, [NO_THROW] * 80)
    assert term and info["outcome"] == "NO_PASS" and r == -1.0 and info["ticks"] <= 500


def test_oracle_pass_dunks(env):
    for seed in range(6):
        scn = sample_scenario("S1" if seed % 2 else "S0", seed)
        sol = oracle.solve(env, scn)
        assert sol["success"], seed
        env.reset(options={"scenario": scn})
        for _ in range((sol["release"] - C.WINDUP_TICKS) // C.DECISION_TICKS):
            env.step(NO_THROW)
        _, r, term, _, info = env.step(sol["action"])  # through the real action decoder
        assert term and info["outcome"] == "DUNK" and r == 1.0, (seed, info)


def test_flat_pass_into_a_defender_is_intercepted(env):
    base = sample_scenario("S1", 4)
    rec = oracle.record(env, base)
    env.reset(options={"scenario": base})
    hold = env.hold
    m = int(np.argmax(rec.dk_state >= 2) + 20)
    target = np.array([*rec.dk_pos[m][:2], 2.2])
    n = 60
    v0, miss = B.solve_release_batch(hold, target[None], np.array([n]))
    mid = B.batch_positions(hold, v0, np.array([n]), np.array([[n // 2]]))[0, 1]
    blocker = DefenderSpec(0, ROLE_STATIC, tuple(mid[:2]), 2.2, 0.6, 0.0, 0.0)
    scn = type(base)(**{**base.__dict__, "defenders": [blocker]})
    env.reset(options={"scenario": scn})
    for _ in range(8):
        env.step(NO_THROW)
    _, r, term, _, info = env.throw(v0[0])
    assert term and info["outcome"] == "INTERCEPTED" and r == -0.5, info


def test_same_seed_same_play(env):
    def run():
        scn = sample_scenario("S1", 9)
        env.reset(options={"scenario": scn})
        rng = np.random.default_rng(3)
        obs_bytes, out = [], None
        for _ in range(60):
            a = rng.uniform(-1, 1, 4)
            a[0] = 1.0 if env.sim.tick >= 40 else -1.0
            out = env.step(a)
            obs_bytes.append(out[0].tobytes())
            if out[2]:
                break
        return obs_bytes, out[1], out[4]["outcome"], out[4]["ticks"]

    assert run() == run()


def test_episode_length_is_bounded(env):
    for seed in range(10):
        scn = sample_scenario("S0", seed)
        env.reset(options={"scenario": scn})
        rng = np.random.default_rng(seed)
        info = {}
        for _ in range(200):
            a = rng.uniform(-1, 1, 4)
            a[0] = 1.0 if rng.random() < 0.05 else -1.0
            _, _, term, _, info = env.step(a)
            if term:
                break
        assert (
            term
            and info["ticks"] <= (C.NO_PASS_TIMEOUT + C.MISS_TIMEOUT) / C.TICK + C.WINDUP_TICKS + 5
        )


def test_native_core_in_env(env):
    assert env.sim.info["native_core"] is True


# ---- rule measures against NumPy references
def _seg_dist(b, axis, lo, hi):
    return math.hypot(math.hypot(b[0] - axis[0], b[1] - axis[1]), max(0.0, lo - b[2], b[2] - hi))


def test_catch_distance_matches_numpy(env):
    rng = np.random.default_rng(0)
    sim = env.sim
    env.reset(options={"scenario": sample_scenario("S0", 0)})
    for _ in range(40):
        dxy = rng.uniform(-6, 6, 2)
        dv = rng.uniform(-6, 6, 2)
        b = np.array([*rng.uniform(-1, 6, 2), rng.uniform(0.2, 4.5)])
        sim.reset(seed=0, initial_state={
            "ball": lobs.Initial(pos=tuple(b), linvel=(0, 0, 0)),
            "dunker": lobs.Initial(q={"jx": dxy[0], "jy": dxy[1]}, qd={"jx": dv[0], "jy": dv[1]}),
        })  # fmt: skip
        fwd = dv / max(np.linalg.norm(dv), 1e-9)
        axis = dxy + C.CATCH_AXIS_FWD * fwd
        want = _seg_dist(b, axis, C.CATCH_Z[0] * C.DUNKER_H_S0, C.CATCH_Z[1] * C.DUNKER_H_S0)
        assert float(sim["catch_d"]) == pytest.approx(want, abs=1e-6)


@pytest.mark.parametrize(
    "pos,vel,event",
    [
        ((3.0, 0.0, 0.1193), (3.0, 0, 0), "ev_floor"),
        ((-0.30, 0.0, 3.4), (-3.0, 0, 0), "ev_board"),
        ((0.2366, 0.0, 3.09), (0, 0, -3.0), "ev_rim"),
        ((13.2, 0.0, 2.0), (3.0, 0, 0), "ev_out"),
    ],
)
def test_world_events_latch_with_their_time(env, pos, vel, event):
    sim = env.sim
    env.reset(options={"scenario": sample_scenario("S0", 0)})
    sim.reset(seed=0, initial_state={"ball": lobs.Initial(pos=pos, linvel=vel)})
    for _ in range(3):
        sim.step()
    assert bool(sim[event])
    assert 0.0 <= float(sim["t_" + event[3:]]) <= 0.03
    for other in ("ev_floor", "ev_board", "ev_rim", "ev_out"):
        if other != event:
            assert not bool(sim[other]), other


def test_free_flight_triggers_nothing(env):
    sim = env.sim
    env.reset(options={"scenario": sample_scenario("S0", 0)})
    sim.reset(seed=0, initial_state={"ball": lobs.Initial(pos=(4.0, 3.0, 5.0), linvel=(0, 0, 3.0))})
    for _ in range(5):
        sim.step()
    assert not any(
        bool(sim[e]) for e in ("ev_floor", "ev_board", "ev_rim", "ev_out", "ev_catch", "ev_blocked")
    )
