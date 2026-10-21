"""Regenerate tests/lobpass/golden/s3a_seed0.npz: a defended play (S3a, seed 0) thrown with a frozen action. The test
replays the same throw and requires the sim trajectory to match to 1e-9 (same pattern as tests/test_golden.py)."""

import numpy as np

from demos.lobpass import oracle
from demos.lobpass.env import LobPassEnv
from demos.lobpass.record import record_episode
from demos.lobpass.scenario import sample_scenario

env = LobPassEnv({"S3a": 1.0})
scn = sample_scenario("S3a", 0)
sol = oracle.solve(env, scn)
assert sol["success"]
arrays, meta = record_episode(scn, "x", 1.0, env, script={"release": sol["release"], "action": sol["action"]})
np.savez_compressed(
    "tests/lobpass/golden/s3a_seed0.npz",
    release=sol["release"], action=np.asarray(sol["action"]), tick=arrays["tick"][::5], pos=arrays["pos"][::5],
    vel=arrays["vel"][::5], dunker_state=arrays["dunker_state"][::5], outcome=meta["outcome"],
)  # fmt: skip
print("golden written:", meta["outcome"], len(arrays["tick"][::5]), "samples")
