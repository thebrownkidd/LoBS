import numpy as np

from demos.lobpass import constants as C
from demos.lobpass import oracle
from demos.lobpass.env import LobPassEnv
from demos.lobpass.scenario import sample_scenario


def test_oracle_success_rate_on_early_stages():
    env = LobPassEnv({"S0": 1.0})
    for stage in ("S0", "S1"):
        seeds = oracle.eval_seeds(stage, 16)
        res = [oracle.solve(env, sample_scenario(stage, s)) for s in seeds]
        assert np.mean([r["success"] for r in res]) >= 0.9, stage
        rep = sum(r["success"] for r in res) / max(sum(r["success_any"] for r in res), 1)
        assert rep >= 0.97


def test_to_action_inverts_decode():
    env = LobPassEnv({"S0": 1.0})
    env.reset(options={"scenario": sample_scenario("S1", 2)})
    a = np.array([1.0, 0.3, -0.2, 0.4])
    v0, clamped = env.decode(a)
    assert not clamped
    assert np.allclose(oracle.to_action(env, v0), a, atol=1e-6)


def test_eval_seeds_are_fixed_and_distinct_per_stage():
    assert oracle.eval_seeds("S0", 3) == [9_000_000, 9_000_001, 9_000_002]
    assert oracle.eval_seeds("S1", 1) == [9_010_000]
    assert C.EVAL_N == 1024
