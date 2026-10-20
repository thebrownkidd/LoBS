import numpy as np
import pytest

import lobs
from lobs import ops
from lobs.errors import LobsError
from lobs.gui.core import bhv
from lobs.gui.core.replay import Replay, lanes_from_sim, segments
from lobs.gui.core.session import Session


def run_world():
    w = lobs.World("r", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_sphere("ball", 0.1, mass=1.0, pos=(0, 0, 1))
    w.add_adversary(
        "gust", on="ball", budget=lobs.Budget(peak=2.0, impulse=0.5), vector=ops.vec(2.0, 0.0, 0.0)
    )
    s = Session(w)
    s.edit("b", lambda sp: bhv.add_behavior(sp, "m", on="ball"))
    s.edit("s", lambda sp: bhv.add_state(sp, "m", "late"))
    s.edit("t", lambda sp: bhv.add_transition(sp, "m", "idle", "late", "ops.time() > 0.4", s.world))
    s.edit("x", lambda sp: sp.measures.update({"x": lobs.props.position("ball").x.node}))
    return s.world


def test_replay_seek_frames_evaluate_and_lanes(tmp_path):
    sim = run_world().simulate("mujoco")
    sim.run(seconds=1.0)
    path = sim.export(tmp_path / "run.npz")
    rp = Replay.load(path)
    assert len(rp) == len(sim.history) and rp.duration == pytest.approx(1.0)
    i = rp.seek_time(0.5)
    assert rp.times[i] == pytest.approx(0.5, abs=0.011)
    live_x = sim.history["x"]
    got = rp.evaluate("props.position(ball).x")
    assert np.allclose(got, live_x)  # replay equals the live run, exactly the library contract
    assert rp.evaluate("props.position(ball).x * 2.0")[-1] == pytest.approx(2 * live_x[-1])
    with pytest.raises(LobsError):  # live-only measures say so, loudly
        rp.evaluate("props.behavior_state('m')")
    lanes = {ln.label: ln for ln in rp.lanes()}
    assert set(lanes) == {"behavior m", "force gust"}
    segs = segments(lanes["behavior m"])
    assert [s[2] for s in segs] == [0, 1] and segs[0][1] == pytest.approx(0.41, abs=0.011)
    assert (
        lanes["force gust"].v.max() == pytest.approx(2.0) and lanes["force gust"].v[-1] == 0.0
    )  # budget spent
    assert lanes["behavior m"].names == ["idle", "late"]
    assert rp.forces_at(0)["gust"].shape == (3,)


def test_replay_advance_loops_and_stops():
    sim = run_world().simulate("mujoco")
    sim.run(seconds=0.3)
    rp = Replay.from_sim(sim)
    rp.loop = False
    while rp.advance(0.05):
        pass
    assert rp.index == len(rp) - 1
    rp.loop = True
    assert rp.advance(0.1) and rp.index < len(rp) - 1


def test_live_lanes_match_trajectory_lanes():
    sim = run_world().simulate("mujoco")
    sim.run(seconds=0.6)
    live = {ln.label: ln for ln in lanes_from_sim(sim)}
    rec = {ln.label: ln for ln in Replay.from_sim(sim).lanes()}
    for k in live:
        assert np.array_equal(live[k].v, rec[k].v)
