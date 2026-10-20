import numpy as np
import pytest

import lobs
from lobs import ops, props
from lobs.errors import LobsError

from .helpers import fake_sim

DT = 0.01
S = [0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0]
T = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]


def sig_world():
    """Two boxes whose heights encode the bool signals S and T, and whose x position is a ramp."""
    w = lobs.World(dt=DT)
    w.add_plane("ground")
    a = w.add_box("a", (0.1, 0.1, 0.1), mass=1.0, pos=(0, 0, 0))
    b = w.add_box("b", (0.1, 0.1, 0.1), mass=1.0, pos=(0, 0, 0))
    w.measure("s", props.height(a) > 0.5)
    w.measure("t", props.height(b) > 0.5)
    w.measure("ramp", props.position(a).x)
    return w, a, b


def motion():
    def f(t):
        i = int(round(t / DT))
        sv, tv = (S[i], T[i]) if i < len(S) else (0, 0)  # the signals are 0 after the pattern ends
        pos = np.array([[i * 0.1, 0.0, float(sv)], [0.0, 0.0, float(tv)]])
        quat = np.tile([1.0, 0, 0, 0], (2, 1))
        return dict(pos=pos, quat=quat, linvel=np.zeros((2, 3)), angvel=np.zeros((2, 3)))

    return f


def run(w, ticks=len(S) - 1, **kw):
    sim = fake_sim(w, motion(), **kw)
    sim.run(ticks=ticks)
    return sim


def hist(sim, name):
    return np.asarray(sim.history[name], dtype=bool)


def ref_rising(x):
    prev, out = False, []
    for v in x:
        out.append(bool(v) and not prev)
        prev = bool(v)
    return out


def test_edges_pulse_once_per_transition():
    w, *_ = sig_world()
    s = ops.ref("s")
    w.measure("rise", s.rising())
    w.measure("fall", s.falling())
    sim = run(w)
    sig = np.array(S, dtype=bool)
    assert list(hist(sim, "rise")) == ref_rising(sig)
    expected_fall = [False] + [
        bool(a) and not bool(b) for a, b in zip(sig[:-1], sig[1:], strict=True)
    ]
    assert list(hist(sim, "fall")) == expected_fall
    assert hist(sim, "rise").sum() == 7  # S has seven separate pulses


def test_true_on_the_first_tick_counts_as_a_rising_edge():
    w, *_ = sig_world()
    w.measure("r", (props.height(w.actor("a")) > -1.0).rising())  # constant True
    sim = run(w, ticks=5)
    assert list(hist(sim, "r")) == [True, False, False, False, False, False]


def test_latch_sets_resets_and_reset_wins():
    w, *_ = sig_world()
    w.measure("held", ops.latch(ops.ref("s"), ops.ref("t")))
    sim = run(w)
    state, ref = False, []
    for s, t in zip(S, T, strict=True):
        state = False if t else (True if s else state)
        ref.append(state)
    assert list(hist(sim, "held")) == ref
    assert any(S[i] and T[i] for i in range(len(S)))  # tick 3 sets and resets at once: reset wins
    w2, *_ = sig_world()
    always = props.height(w2.actor("a")) > -1.0
    w2.measure("both", ops.latch(always, always))
    sim2 = run(w2, ticks=3)
    assert not hist(sim2, "both").any()  # both true on the same tick: reset wins


def test_hold_extends_true_by_seconds_and_retriggers():
    w, *_ = sig_world()
    w.measure("held", ops.ref("s").hold(0.03))
    sim = run(w)
    last, ref = None, []
    for i, s in enumerate(S):
        t = i * DT
        if s:
            last = t
        ref.append(bool(s) or (last is not None and t - last < 0.03 - 1e-9))
    assert list(hist(sim, "held")) == ref


def test_cooldown_ignores_edges_inside_the_window():
    w, *_ = sig_world()
    w.measure("fire", ops.ref("s").cooldown(0.08))
    sim = run(w)
    prev, last, ref = False, None, []
    for i, s in enumerate(S):
        t, edge = i * DT, bool(s) and not prev
        ok = edge and (last is None or t - last >= 0.08 - 1e-9)
        if ok:
            last = t
        ref.append(ok)
        prev = bool(s)
    assert list(hist(sim, "fire")) == ref
    assert (
        hist(sim, "fire").sum() < hist(sim, "fire").size and hist(sim, "fire").sum() < 7
    )  # some edges suppressed


def test_once_fires_only_the_first_edge_and_rearms_on_reset():
    w, *_ = sig_world()
    w.measure("first", ops.ref("s").once())
    sim = run(w)
    assert (
        hist(sim, "first").sum() == 1 and hist(sim, "first")[2]
    )  # tick 2 is the first rising edge
    sim.reset()
    sim.run(ticks=len(S) - 1)
    assert hist(sim, "first").sum() == 1  # armed again after reset


def test_sample_on_captures_the_value_at_the_trigger_edge():
    w, *_ = sig_world()
    w.measure("snap", ops.ref("ramp").sample_on(ops.ref("t")))
    sim = run(w)
    snap, ramp = sim.history["snap"], sim.history["ramp"]
    last = ramp[0]
    for i, tr in enumerate(T):
        if i == 0 or (tr and not (T[i - 1])):
            last = ramp[i] if tr or i == 0 else last
        assert snap[i] == pytest.approx(last), i


def test_reset_clears_stateful_trigger_state():
    w, *_ = sig_world()
    w.measure("held", ops.latch(ops.ref("s"), ops.ref("t")))
    sim = run(w, ticks=2)  # S rises at tick 2: latched
    assert hist(sim, "held")[-1]
    sim.reset()
    assert sim["held"] is False or not bool(sim["held"])  # tick 0 of a fresh run: S[0] = 0


# ------------------------------------------------------------------ randomness
def rand_world():
    w, *_ = sig_world()
    w.measure("coin", ops.chance(0.3))
    w.measure("u", ops.uniform(2.0, 4.0))
    w.measure("n", ops.normal(10.0, 0.5))
    return w


def long_run(w, seed, ticks=3000):
    sim = fake_sim(w, motion(), seed=seed)
    sim.run(ticks=ticks)
    return sim


def test_chance_uniform_normal_statistics_and_seeded_reproducibility():
    w = rand_world()
    a = long_run(w, seed=7)
    coin, u, n = hist(a, "coin"), a.history["u"], a.history["n"]
    assert 0.26 < coin.mean() < 0.34
    assert u.min() >= 2.0 and u.max() <= 4.0 and u.mean() == pytest.approx(3.0, abs=0.06)
    assert n.mean() == pytest.approx(10.0, abs=0.05) and n.std() == pytest.approx(0.5, abs=0.04)
    b = long_run(w, seed=7)
    assert np.array_equal(coin, hist(b, "coin")) and np.array_equal(
        u, b.history["u"]
    )  # same seed, same stream
    c = long_run(w, seed=8)
    assert not np.array_equal(u, c.history["u"])
    a.reset(seed=7)
    a.run(ticks=100)
    assert np.array_equal(a.history["u"], u[:101])  # reset restarts the stream


def test_random_measures_cannot_be_replayed_or_differentiated():
    w = rand_world()
    sim = long_run(w, seed=1, ticks=20)
    traj = sim.trajectory()
    with pytest.raises(LobsError, match="cannot be replayed"):
        traj.evaluate("coin")
    bad = lobs.World(dt=DT)
    bad.add_plane("ground")
    bad.measure("r", ops.uniform(0, 1).rate())
    assert not bad.validate().ok  # rate() of a random quantity is meaningless
    assert np.array_equal(traj.measures["u"], sim.history["u"])  # but the recorded values are kept


def test_sample_on_with_random_gives_one_random_value_per_event():
    w, *_ = sig_world()
    w.measure("delay", ops.uniform(0.1, 0.5).sample_on(ops.ref("s").rising()))
    sim = run(w, seed=3)
    d = sim.history["delay"]
    changes = np.flatnonzero(np.diff(d) != 0) + 1
    edges = [i for i, e in enumerate(ref_rising(np.array(S, bool))) if e and i > 0]
    assert list(changes) == edges  # resampled exactly on the rising edges, constant in between


# ------------------------------------------------------------------ vector helpers
def test_vec_unit_clip_norm():
    w, *_ = sig_world()
    p = props.position(w.actor("a"))
    w.measure("v", ops.vec(p.x, 3.0, 4.0))
    w.measure("u", ops.unit(ops.vec(0.0, 3.0, 4.0)))
    w.measure("zero_unit", ops.unit(ops.vec(0.0, 0.0, 0.0)))
    w.measure("clipped", ops.clip_norm(ops.vec(0.0, 3.0, 4.0), 2.5))
    w.measure("unclipped", ops.clip_norm(ops.vec(0.0, 3.0, 4.0), 9.0))
    sim = run(w, ticks=6)
    assert np.allclose(sim["v"], (0.6, 3.0, 4.0))
    assert (
        np.allclose(sim["u"], (0, 0.6, 0.8))
        and np.allclose(sim["zero_unit"], 0)
        and not np.isnan(sim["zero_unit"]).any()
    )
    assert np.allclose(sim["clipped"], (0, 1.5, 2.0)) and np.allclose(
        sim["unclipped"], (0, 3.0, 4.0)
    )
    bad = lobs.World()
    bad.add_plane("g")
    bad.measure("x", ops.unit(props.height(bad.actor("g"))))  # scalar, not a vector
    assert any("3-vector" in i.message for i in bad.validate().errors)


def test_trigger_inputs_must_be_boolean_and_survive_serialization(tmp_path):
    w, *_ = sig_world()
    w.measure("bad", props.height(w.actor("a")).rising())
    assert any("boolean" in i.message for i in w.validate().errors)
    ok, *_ = sig_world()
    ok.measure("c", ops.latch(ops.ref("s"), ops.ref("t")).hold(0.05).cooldown(0.1))
    ok.save(tmp_path / "t.yaml")
    assert lobs.load_world(tmp_path / "t.yaml").spec == ok.spec
