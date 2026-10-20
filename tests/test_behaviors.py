import numpy as np
import pytest

import lobs
from lobs import act, ops, props, rel
from lobs.errors import LobsError

BACKENDS = lobs.backends.available()


def space(**kw):
    return lobs.World("space", dt=0.002, substeps=5, gravity=(0, 0, 0), **kw)


def sm_one(*do):
    sm = lobs.StateMachine("one")
    sm.state("go", do=list(do))
    return sm


# ---------------------------------------------------------------- Steer
@pytest.mark.parametrize("be", BACKENDS)
def test_arrive_reaches_target_stops_and_respects_max_accel(be):
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=2.0, pos=(0, 0, 1))
    goal = w.anchor("goal", pos=(3, 1, 1))
    w.measure("d", rel.between(me, goal).distance())
    w.measure("v", props.speed(me))
    w.measure("vel", props.velocity(me))
    w.add_behavior(
        "b", on=me, spec=sm_one(act.steer(to=goal, mode="arrive", speed=2.0, max_accel=4.0))
    )
    sim = w.simulate(be)
    sim.run(seconds=8.0)
    assert sim["d"] < 0.05 and sim["v"] < 0.05
    acc = np.linalg.norm(np.diff(sim.history["vel"], axis=0), axis=1) / sim.tick_dt
    assert acc.max() <= 4.0 * 1.05
    assert sim.history["v"].max() <= 2.0 * 1.1


@pytest.mark.parametrize("be", BACKENDS)
def test_flee_increases_distance(be):
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    foe = w.add_box("foe", (0.2, 0.2, 0.2), mass=1.0, pos=(1, 0, 1), fixed=True)
    w.measure("d", rel.between(me, foe).distance())
    w.add_behavior(
        "b", on=me, spec=sm_one(act.steer(to=foe, mode="flee", speed=3.0, max_accel=6.0))
    )
    sim = w.simulate(be)
    sim.run(seconds=2.0)
    assert sim["d"] > 4.0


@pytest.mark.parametrize("be", BACKENDS)
def test_intercept_beats_seek_against_moving_target(be):
    def reach_time(mode):
        w = space()
        me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
        tgt = w.add_box("tgt", (0.2, 0.2, 0.2), mass=1.0, pos=(4, 0, 1))
        w.measure("d", rel.between(me, tgt).distance())
        w.add_behavior(
            "b", on=me, spec=sm_one(act.steer(to=tgt, mode=mode, speed=3.5, max_accel=20.0))
        )
        sim = w.simulate(be)
        sim.reset(initial_state={"tgt": lobs.Initial(linvel=(0, 2.0, 0))})
        sim.run(seconds=6.0)
        d = sim.history["d"]
        hit = np.nonzero(d < 0.4)[0]
        return sim.history.time[hit[0]] if len(hit) else np.inf

    assert reach_time("intercept") < reach_time("seek") - 0.2


def test_steer_needs_mass_and_target():
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1), fixed=True)
    other = w.add_box("o", (0.2, 0.2, 0.2), mass=1.0, pos=(2, 0, 1))
    w.add_behavior("b", on=me, spec=sm_one(act.steer(to=other)))
    assert any("static" in i.message for i in w.validate().errors)
    w2 = space()
    me = w2.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    w2.add_behavior("b", on=me, spec=sm_one(act.steer(mode="seek")))
    assert any("target" in i.message for i in w2.validate().errors)


# ---------------------------------------------------------------- state machines
def machine_world(min_time=0.0):
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    sm = lobs.StateMachine("m")
    sm.var("n", 0.0)
    sm.state("a", on_exit=[act.emit("left_a")])
    sm.state("b", on_enter=[act.set_var("n", props.behavior_var("m", "n") + 1.0)])
    sm.state("c")
    sm.transition("a", "b", when=ops.time() >= 0.3, min_time=min_time)
    sm.transition("b", "c", when=ops.time() >= 0.5)
    w.add_behavior("m", on=me, spec=sm)
    w.measure("n", props.behavior_var("m", "n"))
    w.measure("st", props.behavior_state("m"))
    w.measure("in_b", props.behavior_in_state("m", "b"))
    w.measure("tis", props.behavior_time_in_state("m"))
    w.measure("ev", ops.event("left_a"))
    return w


@pytest.mark.parametrize("be", BACKENDS)
def test_transitions_enter_exit_once_events_one_tick(be):
    sim = machine_world().simulate(be)
    sim.run(seconds=1.0)
    h, t = sim.history, sim.history.time
    assert h["st"][0] == 0 and h["st"][-1] == 2
    assert h["n"][-1] == 1.0  # on_enter ran exactly once
    assert h["ev"].sum() == 1  # Emit is a one-tick event
    k = int(np.nonzero(h["ev"])[0][0])
    assert t[k] == pytest.approx(0.31, abs=1e-9)  # decided at tick 30, visible on the next tick
    assert h["in_b"][np.searchsorted(t, 0.4)]
    assert not h["in_b"][np.searchsorted(t, 0.6)]
    assert h["tis"][np.searchsorted(t, 0.2)] == pytest.approx(0.2, abs=1e-9)
    sim.reset()
    assert sim["st"] == 0 and sim["n"] == 0.0  # variables and state reset


@pytest.mark.parametrize("be", BACKENDS)
def test_min_time_delays_transition(be):
    sim = machine_world(min_time=0.6).simulate(be)
    sim.run(seconds=1.0)
    t = sim.history.time
    first_b = t[np.nonzero(sim.history["st"] == 1)[0][0]]
    assert first_b == pytest.approx(0.61, abs=1e-9)


def test_priority_beats_declaration_order():
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    sm = lobs.StateMachine("p")
    for s in "abc":
        sm.state(s)
    sm.transition("a", "b", when=ops.time() >= 0.1, priority=0)
    sm.transition("a", "c", when=ops.time() >= 0.1, priority=5)
    w.add_behavior("p", on=me, spec=sm)
    w.measure("in_c", props.behavior_in_state("p", "c"))
    sim = w.simulate("mujoco")
    sim.run(seconds=0.3)
    assert sim["in_c"]


# ---------------------------------------------------------------- latency and period
def ramp_world(**kw):
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    step = ops.where(ops.time() >= 0.1, 1.0, 0.0)
    w.add_behavior("b", on=me, spec=sm_one(act.force(ops.vec(step, 0, 0))), **kw)
    w.measure("vx", props.velocity(me).x)
    return w


def first_motion(sim):
    return int(np.nonzero(sim.history["vx"] > 1e-9)[0][0])


@pytest.mark.parametrize("be", BACKENDS)
def test_latency_delays_by_exact_ticks(be):
    a = ramp_world().simulate(be)
    a.run(seconds=0.5)
    b = ramp_world(latency=0.05).simulate(be)
    b.run(seconds=0.5)
    assert first_motion(b) - first_motion(a) == 5


@pytest.mark.parametrize("be", BACKENDS)
def test_period_holds_commands_between_decisions(be):
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    w.add_behavior("b", on=me, spec=sm_one(act.force(ops.vec(ops.time() + 1.0, 0, 0))), period=0.05)
    w.measure("vx", props.velocity(me).x)
    sim = w.simulate(be)
    sim.run(seconds=0.3)
    dv = np.diff(sim.history["vx"]) / sim.tick_dt
    blocks = dv[:25].reshape(5, 5)
    assert np.allclose(blocks, blocks[:, :1], atol=1e-6)  # constant within each decision period
    assert blocks[1, 0] > blocks[0, 0]  # but it does update between decisions


# ---------------------------------------------------------------- scripts
@lobs.behavior("test_wander")
def _wander(ctx):
    ctx.force("me", (ctx.rng.uniform(-1, 1) * ctx.params["scale"], 0.0, 0.0))


@lobs.behavior("test_counter")
class _Counter:
    def reset(self, ctx):
        ctx.state["n"] = 0

    def step(self, ctx):
        ctx.state["n"] += 1
        if ctx.state["n"] == 5:
            ctx.emit("five")


def script_world(scale=1.0):
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    w.add_behavior("wander", on=me, script="test_wander", params={"scale": scale})
    w.add_behavior("count", script="test_counter")
    w.measure("x", props.position(me).x)
    w.measure("five", ops.event("five"))
    return w


@pytest.mark.parametrize("be", BACKENDS)
def test_script_rng_is_reproducible_and_class_form_works(be):
    sim = script_world().simulate(be, seed=3)
    sim.run(seconds=0.5)
    x1 = sim["x"]
    assert sim.history["five"].sum() == 1
    sim.reset(seed=3)
    sim.run(seconds=0.5)
    assert sim["x"] == x1
    sim.reset(seed=4)
    sim.run(seconds=0.5)
    assert sim["x"] != x1


def test_randomize_overrides_script_params():
    sim = script_world(scale=1.0).simulate("mujoco", seed=1)
    sim.reset(seed=1, randomize={"behaviors": {"wander": {"params": {"scale": 0.0}}}})
    sim.run(seconds=0.3)
    assert sim["x"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(LobsError):
        sim.reset(randomize={"behaviors": {"wander": {"nope": 1}}})


def test_unregistered_script_rejected():
    w = space()
    w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    w.add_behavior("b", script="does_not_exist")
    assert any("not registered" in i.message for i in w.validate().errors)


# ---------------------------------------------------------------- ownership
def pend_world():
    w = lobs.World("pend", dt=0.002, substeps=5)
    p = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = p.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0, collision_group=3)
    base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5), damping=0.05),
                   geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5, collision_group=3)  # fmt: skip
    w.spec.contact.disabled_group_pairs.append((3, 3))
    p.add_actuator("motor", "hinge", kind="effort", ctrl_range=(-3.0, 3.0))
    p.set_initial(q={"hinge": 0.4})
    th = w.measure("theta", props.joint_position(p.joint("hinge")))
    w.measure("omega", props.joint_velocity(p.joint("hinge")))
    return w, th


@pytest.mark.parametrize("be", BACKENDS)
def test_owned_actuator_is_driven_and_ignores_external_action(be):
    w, th = pend_world()
    w.add_behavior("pd", on="pend", spec=sm_one(act.actuate("pend/motor", ops.as_unit(-8.0 * props.joint_position("pend/hinge"), "newton*meter") + ops.as_unit(-1.5 * props.joint_velocity("pend/hinge"), "newton*meter"))))  # fmt: skip
    sim = w.simulate(be)
    with pytest.warns(UserWarning, match="owned by a behavior"):
        sim.run(seconds=3.0, controller=lambda s: {"pend/motor": 3.0})
    assert abs(sim["theta"]) < 0.05  # the PD behavior won; the external +3 N m was ignored


@pytest.mark.parametrize("be", BACKENDS)
def test_gym_excludes_owned_actuators(be):
    gymnasium = pytest.importorskip("gymnasium")  # noqa: F841
    w, _ = pend_world()
    w.add_behavior("pd", on="pend", spec=sm_one(act.actuate("pend/motor", 0.0)))
    env = lobs.GymEnv(w, be, observe=["theta"], max_seconds=0.1)
    assert env.action_space.shape == (0,)
    _, info = env.reset(seed=0)
    assert info["behaviors"] == {"pd": "go"}
    with pytest.raises(LobsError):
        lobs.GymEnv(w, be, observe=["theta"], actions=["pend/motor"])


def test_two_owners_rejected():
    w, _ = pend_world()
    w.add_behavior("a", on="pend", spec=sm_one(act.actuate("pend/motor", 1.0)))
    w.add_behavior("b", on="pend", spec=sm_one(act.actuate("pend/motor", 2.0)))
    assert any("also owned" in i.message for i in w.validate().errors)


# ---------------------------------------------------------------- validation, io, determinism
def test_validation_catches_bad_machines():
    w = space()
    me = w.add_box("me", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    sm = lobs.StateMachine("bad")
    sm.state("a")
    sm.state("orphan")
    sm.transition("a", "nowhere", when=ops.time() > 1.0)
    sm.transition("a", "a", when=ops.time())  # not bool
    w.add_behavior("bad", on=me, spec=sm)
    msgs = " | ".join(i.message for i in w.validate().errors)
    assert "unknown state 'nowhere'" in msgs and "bool" in msgs
    assert any("unreachable" in i.message for i in w.validate().warnings)


def test_yaml_round_trip_and_include(tmp_path):
    w = machine_world()
    p = tmp_path / "w.yaml"
    w.save(p)
    w2 = lobs.World.load(p)
    assert w2.spec.behaviors[0].machine.transitions[0].to == "b"
    host = space()
    host.include(w2, prefix="g")
    assert host.spec.behaviors[0].name == "g.m"
    assert host.validate().ok
    sim = host.simulate("mujoco")
    sim.run(seconds=1.0)
    assert sim["g.st"] == 2


@pytest.mark.parametrize("be", BACKENDS)
def test_determinism_and_behavior_order(be):
    def build(order):
        w = space()
        a = w.add_box("a", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
        b = w.add_box("b", (0.2, 0.2, 0.2), mass=1.0, pos=(5, 0, 1))
        specs = {
            "ba": (a, sm_one(act.steer(to=b, mode="seek", speed=2.0))),
            "bb": (b, sm_one(act.steer(to=a, mode="flee", speed=1.0))),
        }
        for k in order:
            on, sm = specs[k]
            w.add_behavior(k, on=on, spec=sm)
        w.measure("pa", props.position(a))
        w.measure("pb", props.position(b))
        return w

    runs = []
    for order in (("ba", "bb"), ("ba", "bb"), ("bb", "ba")):
        s = build(order).simulate(be, seed=5)
        s.run(seconds=1.0)
        runs.append((s.history["pa"].copy(), s.history["pb"].copy()))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    assert np.array_equal(runs[0][0], runs[2][0]) and np.array_equal(runs[0][1], runs[2][1])


def test_behavior_state_is_not_replayable_and_recorded(tmp_path):
    w = machine_world()
    sim = w.simulate("mujoco")
    sim.run(seconds=0.8)
    traj = sim.trajectory()
    assert traj.behaviors["m"][-1] == 2
    p = traj.save(tmp_path / "t.npz")
    assert lobs.Trajectory.load(p).meta["behavior_states"]["m"] == ["a", "b", "c"]
    with pytest.raises(LobsError):
        traj.evaluate(props.behavior_state("m"))


# ---------------------------------------------------------------- generality: the grip recipe
@pytest.mark.parametrize("be", BACKENDS)
def test_grip_recipe_from_primitives(be):
    """Hold a ball at a hand anchor against gravity with only a latch + a triggered spring-damper force."""
    w = lobs.World("grip", dt=0.002, substeps=5, gravity=(0, 0, -9.81))
    hand = w.add_box(
        "hand", (0.1, 0.1, 0.1), mass=5.0, pos=(0, 0, 1.0), fixed=True, collision_group=2
    )
    grip = hand.anchor("grip", pos=(0, 0, 0.0))
    ball = w.add_sphere("ball", 0.05, mass=0.5, pos=(0.0, 0.0, 1.3), collision_group=2)
    w.spec.contact.disabled_group_pairs.append((2, 2))
    d = rel.between(ball, grip).distance()
    near = d < 0.35
    release = (d > 0.5) | (ops.time() >= 1.5)  # lost grip, or let go on a timer
    holding = ops.latch(near.rising(), release.rising())
    k, c = 400.0, 25.0
    spring = ops.as_unit((props.position(grip) - props.position(ball)) * k, "newton") + ops.as_unit(
        (props.velocity(hand) - props.velocity(ball)) * c, "newton"
    )
    w.add_force("grip", on=ball, vector=spring, when=holding, budget=lobs.Budget(peak=200.0))
    w.measure("d", d)
    w.measure("z", props.position(ball).z)
    w.measure("held", holding)
    sim = w.simulate(be)
    sim.run(seconds=1.4)
    assert sim["held"] and sim["d"] < 0.1  # caught and held against gravity
    sim.run(seconds=1.0)
    assert not sim["held"] and sim["z"] < 0.5  # released on the trigger: it falls
    # a disturbance bigger than the peak budget knocks it free
    sim.reset()
    sim.run(seconds=0.6)
    sim.apply_force("ball", (0, 0, 3000.0), ticks=3)
    sim.run(seconds=0.1)
    assert not sim["held"] and sim["d"] > 0.5
