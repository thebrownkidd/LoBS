import numpy as np
import pytest

import lobs
from lobs import math3d as m3
from lobs import ops, props, rel
from lobs.errors import LobsError, WorldValidationError
from lobs.measure import build_evaluator
from lobs.schema.measure import RefNode
from lobs.state import ContactSet

from .helpers import fake_sim

V = np.array([1.0, 0.5, 0.0])
W = 1.2
P0 = np.array([0.0, 0.0, 1.0])
DT = 0.01


def spinner(t):
    return dict(
        pos=(P0 + V * t)[None],
        quat=m3.quat_from_axis_angle([0, 0, 1], W * t)[None],
        linvel=V[None],
        angvel=np.array([[0.0, 0.0, W]]),
    )


def spinner_world():
    w = lobs.World("spin", dt=DT)
    w.add_plane("ground")
    b = w.add_box("b", (1, 1, 1), mass=2.0, pos=tuple(P0))
    tip = b.anchor("tip", pos=(0.5, 0, 0))
    goal = w.anchor("goal", pos=(3, 0, 0))
    return w, b, tip, goal


def tip_state(t):
    r = m3.quat_to_mat(m3.quat_from_axis_angle([0, 0, 1], W * t))
    off = r @ np.array([0.5, 0, 0])
    return P0 + V * t + off, V + np.cross([0, 0, W], off)


def run(w, motion, ticks=60, **kw):
    sim = fake_sim(w, motion, **kw)
    sim.run(ticks=ticks)
    return sim


def test_basic_props_match_analytic():
    w, b, tip, goal = spinner_world()
    w.measure("pos", props.position(b))
    w.measure("h", props.height(b))
    w.measure("v", props.velocity(b))
    w.measure("speed", props.speed(b))
    w.measure("spin", props.spin(b))
    w.measure("heading", props.heading(b))
    w.measure("tilt", props.tilt(b))
    w.measure("tip_v", props.velocity(tip))
    w.measure("tip_speed", props.speed(tip))
    w.measure("mass", props.mass(b))
    sim = run(w, spinner)
    t = sim.history.time
    assert np.allclose(sim.history["pos"], P0 + np.outer(t, V))
    assert np.allclose(sim.history["h"], 1.0)
    assert np.allclose(sim.history["speed"], np.linalg.norm(V))
    assert np.allclose(sim.history["spin"], W)
    assert np.allclose(sim.history["heading"], m3.wrap_angle(W * t))
    assert np.allclose(sim.history["tilt"], 0, atol=1e-12)
    assert sim["mass"] == pytest.approx(2.0)
    exp = np.array([tip_state(x)[1] for x in t])
    assert np.allclose(sim.history["tip_v"], exp, atol=1e-6)
    assert np.allclose(sim.history["tip_speed"], np.linalg.norm(exp, axis=1), atol=1e-6)


def test_distance_and_exact_rate():
    w, b, tip, goal = spinner_world()
    d = w.measure("d", rel.between(tip, goal).distance())
    w.measure("dd", d.rate())
    sim = run(w, spinner)
    for i, t in enumerate(sim.history.time):
        p, v = tip_state(t)
        dvec = np.array([3.0, 0, 0]) - p
        assert sim.history["d"][i] == pytest.approx(np.linalg.norm(dvec), abs=1e-12)
        assert sim.history["dd"][i] == pytest.approx(-(dvec @ v) / np.linalg.norm(dvec), abs=1e-6)


def test_energy_and_momentum():
    w, b, tip, goal = spinner_world()
    w.measure("ke", props.kinetic_energy(b))
    w.measure("pe", props.potential_energy(b))
    w.measure("me", props.mechanical_energy(b))
    w.measure("p", props.momentum(b))
    w.measure("L", props.angular_momentum(b))
    sim = run(w, spinner, ticks=5)
    inertia_zz = 2.0 / 12 * 2
    assert sim["ke"] == pytest.approx(0.5 * 2 * (V @ V) + 0.5 * inertia_zz * W**2)
    assert sim["pe"] == pytest.approx(2.0 * 9.81 * 1.0)
    assert sim["me"] == pytest.approx(sim["ke"] + sim["pe"])
    assert np.allclose(sim["p"], 2 * V)
    assert np.allclose(sim["L"], [0, 0, inertia_zz * W])


def test_rotation_and_axes_angles():
    w, b, tip, goal = spinner_world()
    w.measure("rot", rel.between(b, lobs.WORLD).angle("rotation"))
    w.measure("axes_x", rel.between(b, lobs.WORLD).angle("axes", axis="x"))
    w.measure("axes_z", rel.between(b, lobs.WORLD).angle("axes", axis="z"))
    sim = run(w, spinner)
    t = sim.history.time
    assert np.allclose(sim.history["rot"], np.abs(m3.wrap_angle(W * t)), atol=1e-9)
    assert np.allclose(sim.history["axes_x"], np.abs(m3.wrap_angle(W * t)), atol=1e-9)
    assert np.allclose(sim.history["axes_z"], 0, atol=1e-9)


def orbit_world():
    w = lobs.World("orbit", dt=DT)
    w.add_plane("ground")
    c = w.anchor("c", pos=(0, 0, 0))
    o = w.anchor("o", pos=(0, 0, 0), rule=lobs.Path.circle(1.0, 2.0))
    return w, c, o


def static_motion(t):
    z = np.zeros((0, 3))
    return dict(pos=z, quat=np.zeros((0, 4)), linvel=z, angvel=z)


def test_azimuth_rate_crosses_seam_and_orbit_speed():
    w, c, o = orbit_world()
    az = w.measure("az", rel.between(c, o).angle("azimuth", axis="z", frame="world"))
    w.measure("az_rate", az.rate())
    w.measure("el", rel.between(c, o).angle("elevation", axis="z", frame="world"))
    w.measure("polar", rel.between(c, o).angle("polar", axis="z", frame="world"))
    w.measure("o_speed", props.speed(o))
    sim = run(w, static_motion, ticks=210)
    t = sim.history.time
    assert np.allclose(sim.history["az"], m3.wrap_angle(np.pi * t), atol=1e-9)
    assert np.allclose(
        sim.history["az_rate"], np.pi, atol=1e-5
    )  # no spike at the +-pi seam (t=1, 3)
    assert np.allclose(sim.history["el"], 0, atol=1e-9)
    assert np.allclose(sim.history["polar"], np.pi / 2, atol=1e-9)
    assert np.allclose(sim.history["o_speed"], np.pi, atol=1e-5)


def test_triple_angles_and_recipes():
    w = lobs.World()
    w.add_plane("ground")
    a = w.anchor("a", pos=(1, 0, 0))
    v = w.anchor("v", pos=(0, 0, 0))
    b = w.anchor("b", pos=(0, 1, 0))
    p = w.anchor("p", pos=(1, 0, 2))
    w.measure("inner", rel.between(a, v, b).angle())
    w.measure("signed", rel.between(a, v, b).angle(signed_about="z"))
    w.measure("signed_rev", rel.between(b, v, a).angle(signed_about="z"))
    w.measure("bend", rel.recipes.bend(a, v, b))
    w.measure("curv", rel.recipes.curvature(a, v, b))
    w.measure("along", rel.recipes.along(v, p, "z"))
    w.measure("off", rel.recipes.off_axis(v, p, "z"))
    sim = run(w, static_motion, ticks=1)
    assert sim["inner"] == pytest.approx(np.pi / 2)
    assert sim["signed"] == pytest.approx(np.pi / 2)
    assert sim["signed_rev"] == pytest.approx(-np.pi / 2)
    assert sim["bend"] == pytest.approx(np.pi / 2)
    assert sim["curv"] == pytest.approx(np.sqrt(2))
    assert sim["along"] == pytest.approx(2.0)
    assert sim["off"] == pytest.approx(1.0)


def test_degenerate_geometry_is_nan_not_zero():
    w = lobs.World()
    w.add_plane("ground")
    a = w.anchor("a", pos=(0, 0, 1))
    b = w.anchor("b", pos=(0, 0, 1))
    c = w.anchor("c", pos=(0, 0, 5))  # directly above a: zero azimuth projection
    w.measure("polar_ab", rel.between(a, b).angle("polar"))
    w.measure("az_ac", rel.between(a, c).angle("azimuth", frame="world"))
    w.measure("tri", rel.between(a, b, c).angle())
    w.measure("dist", rel.between(a, b).distance())
    sim = run(w, static_motion, ticks=1)
    assert np.isnan(sim["polar_ab"]) and np.isnan(sim["az_ac"]) and np.isnan(sim["tri"])
    assert sim["dist"] == 0.0


def moving_height(t):
    return dict(
        pos=np.array([[0, 0, 1 + 2 * t]]),
        quat=np.array([[1.0, 0, 0, 0]]),
        linvel=np.array([[0, 0, 2.0]]),
        angvel=np.zeros((1, 3)),
    )


def ramp_world():
    w = lobs.World(dt=DT)
    w.add_plane("ground")
    b = w.add_box("b", (1, 1, 1), mass=1.0, pos=(0, 0, 1))
    return w, b


def test_stateful_time_measures_and_reset():
    w, b = ramp_world()
    h = w.measure("h", props.height(b))
    w.measure("deriv", h.derivative())
    w.measure("integ", h.integral())
    w.measure("win", h.window(0.05).mean())
    w.measure("wmax", h.window(0.05).max())
    w.measure("rmax", h.running_max())
    w.measure("since", h.since_reset())
    w.measure("at0", h.at_reset())
    w.measure("disp", props.displacement(b))
    w.measure("plen", props.path_length(b))
    hi = h > 1.5
    w.measure("hi", hi)
    w.measure("dur", hi.duration_true())
    w.measure("tsince", hi.time_since())
    sim = run(w, moving_height, ticks=100)
    t = sim.history.time
    d = sim.history["deriv"]
    assert np.isnan(d[0]) and np.allclose(d[1:], 2.0, atol=1e-9)
    assert np.allclose(sim.history["integ"], t + t**2, atol=1e-9)
    assert sim.history["win"][-1] == pytest.approx(1 + 2 * (t[-1] - 0.025), abs=1e-9)
    assert sim.history["wmax"][-1] == pytest.approx(1 + 2 * t[-1])
    assert np.allclose(sim.history["since"], 2 * t) and np.allclose(sim.history["at0"], 1.0)
    assert np.allclose(sim.history["disp"][:, 2], 2 * t)
    assert np.allclose(sim.history["plen"], 2 * t, atol=1e-9)
    first_true = t[sim.history["hi"]][0]
    assert first_true == pytest.approx(0.26)
    assert sim.history["dur"][-1] == pytest.approx(t[-1] - first_true)
    assert np.isnan(sim.history["tsince"][0]) and sim.history["tsince"][-1] == pytest.approx(
        t[-1] - first_true
    )
    assert sim.history["dur"][0] == 0.0

    sim.reset()  # no stale values after reset
    assert len(sim.history) == 1
    assert sim["integ"] == 0.0 and np.isnan(sim["deriv"]) and sim["plen"] == 0.0
    assert np.allclose(sim["disp"], 0) and sim["dur"] == 0.0 and np.isnan(sim["tsince"])


def test_operators_and_literals():
    w, b = ramp_world()
    h = w.measure("h", props.height(b))
    w.measure("twice", h * 2)
    w.measure("minus", 5 - h)
    w.measure("ratio", h / 2.0)
    w.measure("sq", h**2)
    w.measure("clipped", ops.clip(h, hi=1.5))
    w.measure("sel", ops.where(h > 1.1, h, -h))
    w.measure("mx", ops.max_of(h, h * 2, 0.0 * h))
    w.measure("nrm", props.position(b).norm())
    w.measure("pz", props.position(b).z)
    w.measure("dotz", ops.dot(props.position(b), [0, 0, 1.0]))
    w.measure("both", (h > 1.1) & (h < 1.3))
    w.measure("either", (h < 1.1) | (h > 1.3))
    w.measure("neg", ~(h > 1.2))
    w.measure("eq", h.eq(1.0, tol=1e-9))
    sim = run(w, moving_height, ticks=10)
    x = 1 + 2 * 0.1
    assert sim["twice"] == pytest.approx(2 * x) and sim["minus"] == pytest.approx(5 - x)
    assert sim["ratio"] == pytest.approx(x / 2) and sim["sq"] == pytest.approx(x * x)
    assert sim["clipped"] == pytest.approx(1.2) and sim["sel"] == pytest.approx(x)
    assert sim["mx"] == pytest.approx(2 * x) and sim["nrm"] == pytest.approx(x)
    assert sim["pz"] == pytest.approx(x) and sim["dotz"] == pytest.approx(x)
    assert sim["both"] is False or sim["both"] == (1.1 < x < 1.3)
    assert sim["either"] == (x < 1.1 or x > 1.3)
    assert sim["neg"] == (not x > 1.2) and sim["eq"] is False


def errors_for(build) -> str:
    w, b = ramp_world()
    build(w, b)
    rep = w.validate()
    assert not rep.ok
    return " ".join(i.message for i in rep.errors)


def test_validation_errors():
    assert "unit mismatch" in errors_for(
        lambda w, b: w.measure("bad", props.height(b) + props.speed(b))
    )
    assert "rate" in errors_for(lambda w, b: w.measure("bad", props.speed(b).rate()))
    assert "unknown path" in errors_for(lambda w, b: w.measure("bad", props.height("nope")))
    assert "expected" in errors_for(lambda w, b: w.measure("bad", props.joint_position(b)))
    assert "numeric" in errors_for(lambda w, b: w.measure("bad", props.orientation(b) + 1.0))
    assert "numeric" in errors_for(lambda w, b: w.measure("bad", (props.height(b) > 1) + 1.0))
    assert "needs a kind" in errors_for(
        lambda w, b: w.measure("bad", rel.between(b, lobs.WORLD).angle("bogus"))
    )
    assert "boolean" in errors_for(lambda w, b: w.measure("bad", props.height(b) & props.height(b)))
    assert "unknown property" in errors_for(
        lambda w, b: w.spec.measures.__setitem__(
            "bad", props.height(b).node.model_copy(update={"prop": "zzz"})
        )
    )
    assert "not registered" in errors_for(
        lambda w, b: w.measure("bad", ops.custom("no_such_fn", {"x": props.height(b)}))
    )

    def cycle(w, b):
        w.spec.measures["a"] = RefNode(name="b")
        w.spec.measures["b"] = RefNode(name="a")

    assert "cycle" in errors_for(cycle)
    w, b = ramp_world()
    w.measure("bad", props.height(b) + props.speed(b))
    with pytest.raises(WorldValidationError):
        fake_sim(w, moving_height)
    with pytest.raises(TypeError):
        bool(props.height(b) > 1)
    with pytest.raises(LobsError):
        w.measure("bad", props.height(b))  # duplicate name
    with pytest.raises(LobsError):
        w.measure("1abc", props.height(b))


def test_dedup_and_numpy_scalar_interop():
    w, b = ramp_world()
    w.measure("h1", props.height(b))
    w.measure("h2", props.height(b))
    ev = build_evaluator(w.registry(), w.spec.measures)
    assert ev.outputs["h1"] == ev.outputs["h2"]
    m = np.float64(2.0) * props.height(b)  # numpy scalar on the left defers to Measure
    assert isinstance(m, lobs.measure.Measure)


def test_joint_measures_and_rate_of_joint():
    w = lobs.World(dt=DT)
    pend = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = pend.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0)
    base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5), range=(-1, 1)),
                   geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5)  # fmt: skip
    j = pend.joint("hinge")
    q = w.measure("q", props.joint_position(j))
    w.measure("qd", props.joint_velocity(j))
    w.measure("margin", props.joint_limit_margin(j))
    w.measure("q_rate", q.rate())
    w.measure("tip_h", props.height(pend.body("rod")))

    def motion(t):
        ang = 0.5 * np.sin(t)
        pos = np.array([[-0.5 * np.sin(ang), 0, 2 - 0.5 * np.cos(ang)]])
        return dict(pos=pos, quat=m3.quat_from_axis_angle([0, 1, 0], ang)[None], linvel=np.zeros((1, 3)),
                    angvel=np.zeros((1, 3)), q=np.array([ang]), qd=np.array([0.5 * np.cos(t)]))  # fmt: skip

    sim = run(w, motion, ticks=100)
    t = sim.history.time
    assert np.allclose(sim.history["q"], 0.5 * np.sin(t))
    assert np.allclose(sim.history["qd"], 0.5 * np.cos(t))
    assert np.allclose(sim.history["q_rate"], 0.5 * np.cos(t), atol=1e-6)
    assert np.allclose(sim.history["margin"], 1 - 0.5 * np.abs(np.sin(t)))


def test_contact_props_with_fake_contacts():
    w, b = ramp_world()
    w.measure("touch", props.in_contact(b))
    w.measure("touch_ground", props.in_contact(b, with_=w.actor("ground")))
    w.measure("n", props.contact_count(b))
    w.measure("f", props.contact_force(b))

    def contacts(t):
        return ContactSet(
            body_a=np.array([0, 0]), body_b=np.array([1, 1]), point=np.zeros((2, 3)),
            normal=np.tile([0, 0, 1.0], (2, 1)), depth=np.array([1e-3, 1e-3]),
            force_on_b=np.array([[0, 0, 5.0], [0, 0, 4.0]]), normal_force=np.array([5.0, 4.0]),
        )  # fmt: skip

    sim = run(w, moving_height, ticks=2, contacts=contacts)
    assert sim["touch"] is True and sim["touch_ground"] is True
    assert sim["n"] == 2 and sim["f"] == pytest.approx(9.0)
    sim2 = run(w, moving_height, ticks=2)  # no contacts -> empty set
    assert sim2["touch"] is False and sim2["f"] == 0.0


def test_custom_function_roundtrip(tmp_path):
    @lobs.register_function("double_it")
    def double_it(x):
        return 2.0 * x

    w, b = ramp_world()
    w.measure("c", ops.custom("double_it", {"x": props.height(b)}, unit="meter"))
    w.save(tmp_path / "w.yaml")
    sim = run(lobs.load_world(tmp_path / "w.yaml"), moving_height, ticks=1)
    assert sim["c"] == pytest.approx(2 * (1 + 2 * DT))


def test_replay_trajectory_and_posthoc_anchor(tmp_path):
    w, b, tip, goal = spinner_world()
    d = w.measure("d", rel.between(tip, goal).distance())
    w.measure("dd", d.rate())
    sim = run(w, spinner, ticks=40)
    traj = sim.trajectory()
    p = sim.export(tmp_path / "run.npz")
    loaded = lobs.Trajectory.load(p)
    assert np.allclose(loaded.t, sim.history.time)
    assert np.allclose(loaded.evaluate("d"), sim.history["d"])
    assert np.allclose(loaded.evaluate("dd"), sim.history["dd"], atol=1e-9)
    assert loaded.meta["measures"]["d"]["unit"] == "meter"
    assert loaded.meta["measures"]["dd"]["provenance"] == "computed"
    # new anchor + new measure on an old run, no re-simulation
    late = loaded.world.anchor("late", pos=(0, 5, 0))
    got = loaded.evaluate(rel.between(loaded.world.actor("b"), late).distance())
    exp = np.linalg.norm(np.array([0, 5, 0]) - (P0 + np.outer(loaded.t, V)), axis=1)
    assert np.allclose(got, exp)
    assert traj.meta["backend"]["engine"] == "fake"


def test_history_limit_and_record_every():
    w, b = ramp_world()
    w.measure("h", props.height(b))
    sim = run(w, moving_height, ticks=50, history_limit=10, record_every=1)
    assert len(sim.history) == 10 and sim.history.time[-1] == pytest.approx(0.5)
    sim2 = run(w, moving_height, ticks=50, record_every=5)
    assert len(sim2.history) == 11  # ticks 0,5,...,50
    assert sim2["h"] == pytest.approx(1 + 2 * 0.5)  # latest value is always current
