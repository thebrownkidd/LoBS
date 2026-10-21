"""The thin tooling around the demo: the studio brain, the evaluator's summary and the oracle-free pieces."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from demos.lobpass import constants as C  # noqa: E402
from demos.lobpass import evaluate, studio  # noqa: E402
from demos.lobpass.ppo import Policy  # noqa: E402
from demos.lobpass.train import load_policy, save_ckpt  # noqa: E402


@pytest.fixture()
def always_throw_ckpt(tmp_path):
    """A policy that throws on its first decision, with a sensible lob (actions near the middle of the box)."""
    torch.manual_seed(0)
    pol = Policy()
    with torch.no_grad():
        pol.actor[-1].bias.copy_(torch.tensor([6.0, 0.0, -0.2, 0.0]))
        pol.actor[-1].weight.mul_(0.0)
    opt = torch.optim.Adam(pol.parameters())
    path = tmp_path / "always.pt"
    save_ckpt(path, pol, opt, "S0", 0, 0, 0, {"ppo": {"hidden": C.HIDDEN}})
    return str(path)


def test_checkpoint_loader_rejects_a_different_observation_layout(tmp_path):
    pol = Policy()
    path = tmp_path / "bad.pt"
    save_ckpt(path, pol, torch.optim.Adam(pol.parameters()), "S0", 0, 0, 0, {"ppo": {"hidden": C.HIDDEN}})
    ck = torch.load(path, weights_only=False)
    ck["obs_version"] = "deadbeef"
    torch.save(ck, path)
    with pytest.raises(AssertionError):
        load_policy(path)


def test_studio_brain_throws_in_a_script_behavior_world(always_throw_ckpt):
    res = studio.check(always_throw_ckpt, "S1", 5, seconds=3.0)
    assert res["released"] is True
    assert res["native_core"] is False  # a script behavior moves the world onto the Python tick path


def test_studio_world_round_trips_through_yaml(always_throw_ckpt, tmp_path):
    w = studio.build_show_world(always_throw_ckpt, "S2", 11)
    path = tmp_path / "show.yaml"
    w.save(path)
    import lobs

    again = lobs.load_world(path)
    assert [b.name for b in again.spec.behaviors].count("brain") == 1
    assert w.validate().ok


def test_evaluate_summary_breaks_outcomes_down():
    infos = [
        {"outcome": "DUNK", "n_def": 1, "finish": "air", "scenario_seed": 1},
        {"outcome": "DUNK", "n_def": 1, "finish": "ground", "scenario_seed": 2},
        {"outcome": "MISS", "n_def": 2, "miss_kind": "rim", "miss_d": 0.7, "scenario_seed": 3},
        {"outcome": "INTERCEPTED", "n_def": 2, "scenario_seed": 4},
    ]
    ev = {"infos": infos, "success": 0.5, "outcomes": {"DUNK": 0.5, "MISS": 0.25, "INTERCEPTED": 0.25}, "reward": 0.1,
          "release_time": 1.9, "clamp_rate": 0.0}  # fmt: skip
    r = evaluate.summarize(ev, "S2")
    assert r["dunk_by_defenders"] == {"1": 1.0, "2": 0.0}
    assert r["finish"] == {"air": 0.5, "ground": 0.5}
    assert r["miss_kind"] == {"rim": 1.0}
    assert r["failure_examples"] == {"MISS": [3], "INTERCEPTED": [4]}
    assert np.isclose(r["mean_miss_distance"], 0.7)
