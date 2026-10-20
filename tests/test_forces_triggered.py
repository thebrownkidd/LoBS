import numpy as np
import pytest

import lobs
from lobs import ops, props
from lobs.errors import LobsError

BACKENDS = lobs.backends.available()


def free_box(m=2.0, v0=(0, 0, 0)):
    w = lobs.World(dt=0.002, substeps=5, gravity=(0, 0, 0))
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=m, pos=(0, 0, 1))
    return w, box


@pytest.mark.parametrize("be", BACKENDS)
def test_when_gates_force_on_exact_ticks(be):
    w, box = free_box()
    w.add_force("push", on=box, magnitude=4.0, direction=(1, 0, 0), when=ops.time() >= 0.1)
    w.measure("vx", props.velocity(box).x)
    w.measure("act", props.force_active("push"))
    sim = w.simulate(be)
    sim.run(seconds=0.05)
    assert sim["vx"] == pytest.approx(0.0, abs=1e-9)
    assert not sim["act"]
    sim.run(seconds=0.25)  # 0.3 s total, force active from 0.1 s on: 0.2 s * 2 m/s^2
    assert sim["vx"] == pytest.approx(0.4, rel=0.06)
    assert sim["act"]


@pytest.mark.parametrize("be", BACKENDS)
def test_vector_drag_decays_exponentially(be):
    m, k, v0 = 2.0, 1.5, 3.0
    w, box = free_box(m)
    w.add_force("drag", on=box, vector=props.velocity(box) * -k)
    w.measure("vx", props.velocity(box).x)
    sim = w.simulate(be)
    sim.reset(initial_state={"box": lobs.Initial(linvel=(v0, 0, 0))})
    sim.run(seconds=1.0)
    # zero-order hold of drag is first-order accurate in dt
    assert sim["vx"] == pytest.approx(v0 * np.exp(-k * sim.time / m), rel=2e-2)


@pytest.mark.parametrize("be", BACKENDS)
def test_peak_clamp(be):
    w, box = free_box(1.0)
    w.add_force("big", on=box, magnitude=100.0, direction=(1, 0, 0), budget=lobs.Budget(peak=3.0))
    w.measure("vx", props.velocity(box).x)
    sim = w.simulate(be)
    sim.run(seconds=1.0)
    assert sim["vx"] == pytest.approx(3.0, rel=1e-2)
    assert sim.force_stats["big"]["peak"] == pytest.approx(3.0)


@pytest.mark.parametrize("be", BACKENDS)
def test_impulse_budget_exhausted_exactly_then_silent(be):
    w, box = free_box(1.0)
    w.add_force(
        "shove", on=box, magnitude=10.0, direction=(1, 0, 0), budget=lobs.Budget(impulse=0.7)
    )
    w.measure("vx", props.velocity(box).x)
    w.measure("imp", props.force_impulse("shove"))
    sim = w.simulate(be)
    sim.run(seconds=0.5)
    assert sim["imp"] == pytest.approx(0.7, abs=1e-6)
    v = sim["vx"]
    assert v == pytest.approx(0.7, rel=1e-2)
    sim.run(seconds=0.5)
    assert sim["vx"] == pytest.approx(v, abs=1e-6)
    assert (
        not sim.force_stats["shove"]["active"]
        or np.linalg.norm(sim.force_stats["shove"]["last_force"]) == 0
    )


@pytest.mark.parametrize("be", BACKENDS)
def test_active_time_budget_and_reset(be):
    w, box = free_box(1.0)
    w.add_force(
        "f", on=box, magnitude=1.0, direction=(1, 0, 0), budget=lobs.Budget(active_time=0.2)
    )
    sim = w.simulate(be)
    sim.run(seconds=1.0)
    assert sim.force_stats["f"]["active_time"] == pytest.approx(0.2, abs=sim.tick_dt)
    sim.reset()
    st = sim.force_stats["f"]
    assert st["impulse"] == 0 and st["active_time"] == 0 and st["peak"] == 0


@pytest.mark.parametrize("be", BACKENDS)
def test_reactive_adversary_pushes_only_inside_region(be):
    w = lobs.World(dt=0.002, substeps=5, gravity=(0, 0, 0))
    ball = w.add_sphere("ball", 0.1, mass=1.0, pos=(0, 0, 1))
    w.add_adversary(
        "gust", on=ball, budget=lobs.Budget(peak=2.0),
        vector=ops.vec(0, 2.0, 0), when=props.position(ball).x > 0.5,
    )  # fmt: skip
    w.measure("y", props.position(ball).y)
    sim = w.simulate(be)
    sim.reset(initial_state={"ball": lobs.Initial(linvel=(1.0, 0, 0))})
    sim.run(seconds=0.4)  # still at x < 0.5
    assert sim["y"] == pytest.approx(0.0, abs=1e-9)
    sim.run(seconds=1.0)
    assert sim["y"] > 0.1
    y = sim["y"]
    sim.reset(initial_state={"ball": lobs.Initial(linvel=(1.0, 0, 0))})
    sim.run(seconds=1.4)
    assert sim["y"] == pytest.approx(y, abs=1e-9)  # deterministic


def test_adversary_requires_peak():
    w, box = free_box()
    with pytest.raises(LobsError):
        w.add_adversary(
            "x", on=box, budget=lobs.Budget(impulse=1.0), magnitude=1.0, direction=(1, 0, 0)
        )


def test_force_spec_errors_and_hidden_measures(tmp_path):
    w, box = free_box()
    w.add_force("both", on=box, magnitude=1.0, direction=(1, 0, 0), vector=ops.vec(1, 0, 0))
    assert not w.validate().ok
    w2, box = free_box()
    w2.add_force("v", on=box, vector=props.speed(box))  # scalar, not a vector
    assert any("vector" in i.message for i in w2.validate().errors)
    w3, box = free_box()
    w3.add_force("g", on=box, magnitude=1.0, direction=(1, 0, 0), when=props.speed(box))  # not bool
    assert any("when" in i.message for i in w3.validate().errors)
    w4, box = free_box()
    w4.add_force("a", on=box, magnitude=1.0, direction=(1, 0, 0), adversary=True)
    assert any("budget.peak" in i.message for i in w4.validate().warnings)


def test_yaml_round_trip_and_include(tmp_path):
    w, box = free_box()
    w.add_force(
        "gust", on=box, vector=ops.vec(0, 1, 0) * props.speed(box), when=ops.time() > 0.1,
        budget=lobs.Budget(peak=2.0, impulse=1.0), adversary=True,
    )  # fmt: skip
    path = tmp_path / "w.yaml"
    w.save(path)
    w2 = lobs.World.load(path)
    f = w2.spec.forces[0]
    assert f.adversary and f.budget.peak == 2.0 and f.vector is not None and f.when is not None
    host = lobs.World("host", dt=0.002, substeps=5, gravity=(0, 0, 0))
    host.include(w2, prefix="g")
    assert host.spec.forces[0].name == "g.gust"
    assert host.validate().ok


@pytest.mark.parametrize("be", BACKENDS)
def test_forces_recorded_and_hidden(be, tmp_path):
    w, box = free_box(1.0)
    w.add_force("push", on=box, magnitude=2.0, direction=(1, 0, 0))
    sim = w.simulate(be, record=True)
    sim.run(ticks=10)
    assert not any(n.startswith("__force") for n in sim.history.names())
    traj = sim.trajectory()
    assert traj.forces["push"].shape == (len(traj), 3)
    assert traj.forces["push"][-1][0] == pytest.approx(2.0)
    p = traj.save(tmp_path / "t.npz")
    assert lobs.Trajectory.load(p).forces["push"].shape == traj.forces["push"].shape
