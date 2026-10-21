"""Animation QA on five plays: an alley-oop dunk, a dunk off a catch on the run, an interception, a miss and a block.
Each is recorded from the real LoBS world, baked, and measured against the B10 limits."""

import dataclasses

import numpy as np
import pytest

from demos.lobpass import ballistics as B
from demos.lobpass import constants as C
from demos.lobpass import oracle
from demos.lobpass.anim import bake, camera, qa
from demos.lobpass.env import LobPassEnv
from demos.lobpass.record import record_episode
from demos.lobpass.scenario import ROLE_STATIC, DefenderSpec, sample_scenario

POST = 1.2  # seconds recorded after the outcome


@pytest.fixture(scope="module")
def env():
    return LobPassEnv({"S0": 1.0})


def bake_and_check(arrays, meta):
    baked = bake.bake_episode(arrays, meta)
    direction = camera.direct(baked)
    return baked, qa.check(baked, direction)


def assert_clean(res):
    bad = qa.failures(res)
    assert not bad, "\n" + qa.report(res)


def _alley_oop(env):
    return record_episode(sample_scenario("S1", 3), "oracle", POST, env)


def _ground_dunk(env):
    for seed in range(60):
        scn = sample_scenario("S1", seed)
        sol = oracle.solve(env, scn, kinds=(0,))
        if sol["success"]:
            return record_episode(
                scn, "x", POST, env, script={"release": sol["release"], "action": sol["action"]}
            )
    raise AssertionError("no ground-catch dunk found")


def _intercepted(env):
    base = sample_scenario("S1", 4)
    rec = oracle.record(env, base)
    env.reset(options={"scenario": base})
    m = int(np.argmax(rec.dk_state >= 2) + 20)
    v0, _ = B.solve_release_batch(env.hold, np.array([[*rec.dk_pos[m][:2], 2.2]]), np.array([60]))
    mid = B.batch_positions(env.hold, v0, np.array([60]), np.array([[30]]))[0, 1]
    blocker = DefenderSpec(0, ROLE_STATIC, (float(mid[0]), float(mid[1])), 2.2, 0.6, 0.0, 0.0)
    scn = dataclasses.replace(base, defenders=[blocker])
    return record_episode(scn, "x", POST, env, script={"release": 8 * 10 + 15, "v0": v0[0]})


def _miss(env):
    scn = sample_scenario("S1", 3)
    sol = oracle.solve(env, scn)
    a = np.array(sol["action"])
    a[1] = np.clip(a[1] + 0.9, -1, 1)
    return record_episode(scn, "x", POST, env, script={"release": sol["release"], "action": a})


def _blocked(env):
    """The throw is committed just before the dunker runs into a defender: the block comes first, so BLOCKED."""
    base = dataclasses.replace(sample_scenario("S1", 4), corner=None, start_delay=0.0)
    start, T = np.array(base.dunker_start), np.array(base.takeoff)
    p = start + 0.6 * (T - start)
    blocker = DefenderSpec(0, ROLE_STATIC, (float(p[0]), float(p[1])), 2.0, 0.7, 0.0, 0.0)
    scn = dataclasses.replace(base, defenders=[blocker])
    rec = oracle.record(env, scn)
    t_block = int(np.argmax(rec.dk_state == 6))
    decision = max((t_block - 8) // 10 * 10, 0)
    return record_episode(
        scn, "x", POST, env, script={"release": decision + C.WINDUP_TICKS, "v0": [2.0, 0.0, 6.0]}
    )


CASES = {
    "alley_oop": (_alley_oop, "DUNK"),
    "ground_dunk": (_ground_dunk, "DUNK"),
    "intercepted": (_intercepted, "INTERCEPTED"),
    "miss": (_miss, "MISS"),
    "blocked": (_blocked, "BLOCKED"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_animation_qa(env, name):
    make, outcome = CASES[name]
    arrays, meta = make(env)
    assert meta["outcome"] == outcome, meta["outcome"]
    baked, res = bake_and_check(arrays, meta)
    assert_clean(res)
    assert np.isfinite(baked.ball).all() and all(
        np.isfinite(t["q"]).all() for t in baked.tracks.values()
    )


def test_ball_leaves_the_hand_at_the_sim_position():
    """At the release tick the visual ball is exactly the sim ball (the hold point)."""
    env = LobPassEnv({"S0": 1.0})
    arrays, meta = _alley_oop(env)
    baked = bake.bake_episode(arrays, meta)
    f = baked.plan.f_release
    assert np.allclose(baked.ball[f], baked.sim.pos["ball"][f], atol=1e-9)


def test_video_export_roundtrip(tmp_path, env):
    arrays, meta = _alley_oop(env)
    baked = bake.bake_episode(arrays, meta)
    direction = camera.direct(baked)
    out = bake.export(baked, direction, tmp_path / "anim.json")
    import json

    d = json.loads(out.read_text())
    assert d["format"] == "lobpass-anim" and d["fps"] == 60 and len(d["characters"]) == 2
    assert d["slow"] is not None and d["frames"] > 200
    assert any(e["type"] == "slam" for e in d["events"])
    assert C.TICK > 0
