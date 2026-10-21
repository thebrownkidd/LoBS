import pathlib

import numpy as np
import pytest

from demos.lobpass.env import LobPassEnv
from demos.lobpass.record import record_episode
from demos.lobpass.scenario import sample_scenario

GOLDEN = pathlib.Path(__file__).parent / "golden" / "s3a_seed0.npz"


@pytest.mark.skipif(not GOLDEN.exists(), reason="run tools/make_lobpass_golden.py")
def test_defended_play_reproduces_the_golden_trajectory():
    g = np.load(GOLDEN)
    env = LobPassEnv({"S3a": 1.0})
    arrays, meta = record_episode(sample_scenario("S3a", 0), "x", 1.0, env, script={"release": int(g["release"]), "action": g["action"]})
    assert meta["outcome"] == str(g["outcome"]) == "DUNK"
    assert np.array_equal(arrays["tick"][::5], g["tick"])
    assert np.allclose(arrays["pos"][::5], g["pos"], atol=1e-9, rtol=0)
    assert np.allclose(arrays["vel"][::5], g["vel"], atol=1e-9, rtol=0)
    assert np.array_equal(arrays["dunker_state"][::5], g["dunker_state"])
