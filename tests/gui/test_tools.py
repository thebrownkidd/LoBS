import os
import time

import numpy as np
import pytest

import lobs
from lobs import ops
from lobs.gui.core import tools
from lobs.gui.core.session import Session


def ball_world():
    w = lobs.World("mc", dt=0.002, substeps=5)
    w.add_box("floor", (20, 20, 0.2), fixed=True, pos=(0, 0, -0.1))
    w.add_sphere("ball", 0.1, mass=0.5, pos=(0, 0, 1.0))
    w.measure("z", lobs.props.position("ball").z)
    w.measure("x", lobs.props.position("ball").x)
    return w


def test_check_and_compare_wrappers():
    res = tools.run_check(ball_world(), "mujoco", 1.0)
    assert res.passed, res.summary()
    if {"mujoco", "chrono"} <= set(lobs.backends.available()):
        comp = tools.run_compare(ball_world(), ("mujoco", "chrono"), 0.5)
        assert "z" in comp.table()


def test_monte_carlo_is_reproducible_and_bins_map_back_to_seeds():
    rnd = {
        "ball": {
            "pos": (lobs.Uniform(-1, 1), 0.0, 1.0),
            "linvel": (lobs.Normal(0.0, 0.5), 0.0, 0.0),
        }
    }
    a = tools.monte_carlo(ball_world(), "x", runs=8, seconds=0.5, randomize=rnd, stat="final")
    b = tools.monte_carlo(ball_world(), "x", runs=8, seconds=0.5, randomize=rnd, stat="final")
    assert np.array_equal(a["values"], b["values"]) and len(set(a["values"])) > 4
    counts, edges = a["histogram"]
    assert counts.sum() == 8
    covered = sorted(s for i in range(len(counts)) for s in tools.seeds_in_bin(a, i))
    assert covered == a["seeds"]
    # replaying one seed reproduces its recorded outcome
    seed = a["seeds"][3]
    sim = ball_world().simulate("mujoco")
    sim.reset(seed=seed, randomize=rnd)
    sim.run(seconds=0.5)
    assert float(sim["x"]) == pytest.approx(a["values"][3])
    with pytest.raises(lobs.errors.LobsError):
        tools.monte_carlo(ball_world(), "x", stat="median")


def test_console_runs_statements_keeps_names_and_reports_errors():
    w = ball_world()
    sim = w.simulate("mujoco")
    c = tools.Console(lambda: w, lambda: sim)
    assert c.run("1 + 2").strip() == "3"
    c.run("k = 5")
    assert c.run("k * 2").strip() == "10"
    sim.run(ticks=10)
    assert float(c.run("float(sim['z'])")) < 1.0
    assert "ZeroDivisionError" in c.run("1 / 0")
    assert c.run("print('hi')").strip() == "hi"
    assert "props" in c.namespace() and "world" in c.namespace()


def test_autosave_and_recovery(tmp_path):
    s = Session(ball_world())
    path = tmp_path / "w.yaml"
    s.save(path)
    assert tools.write_autosave(s) is None  # nothing dirty yet
    s.edit("g", lambda sp: setattr(sp, "gravity", (0.0, 0.0, -2.0)))
    auto = tools.write_autosave(s)
    assert auto is not None and auto.exists()
    os.utime(
        path, (time.time() - 100, time.time() - 100)
    )  # the saved file is older than the autosave
    assert tools.recoverable(path) == auto
    assert Session.open(auto).world.spec.gravity == (0.0, 0.0, -2.0)
    tools.clear_autosave(path)
    assert tools.recoverable(path) is None
    assert ops
