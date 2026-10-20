"""The native tick core must match the Python runtime on the behavior / force-law features."""

import numpy as np
import pytest

import lobs
from lobs import act, native, ops, props, rel

pytestmark = pytest.mark.skipif(
    not native.available(), reason="native kernel not built (python tools/build_native.py)"
)


def run_both(build, seconds, controller=None, seed=3):
    out = {}
    import os

    for mode in ("python", "native"):
        os.environ["LOBS_NATIVE_CORE"] = "0" if mode == "python" else "1"
        try:
            sim = build().simulate("mujoco", seed=seed)
        finally:
            os.environ.pop("LOBS_NATIVE_CORE", None)
        assert (sim._core is not None) == (mode == "native")
        sim.run(seconds=seconds, controller=controller)
        out[mode] = sim
    return out["python"], out["native"]


def assert_same(a, b, rtol=1e-9, atol=1e-11):
    for n in a.history.names():
        assert np.allclose(a.history[n], b.history[n], rtol=rtol, atol=atol, equal_nan=True), n
    assert np.allclose(a.state().body_pos, b.state().body_pos, rtol=rtol, atol=atol)
    for k, v in a.force_stats.items():
        w = b.force_stats[k]
        assert v["active"] == w["active"] and np.isclose(
            v["impulse"], w["impulse"], rtol=rtol, atol=atol
        )
        assert np.allclose(v["last_force"], w["last_force"], rtol=rtol, atol=atol)


def machine_world():
    w = lobs.World("m", dt=0.002, substeps=5, gravity=(0, 0, 0))
    ball = w.add_sphere("ball", 0.1, mass=0.5, pos=(-3, 0, 1), linvel=(2.0, 0.2, 0))
    a = w.add_box("a", (0.4, 0.4, 0.8), mass=60.0, pos=(1, 2, 1))
    b = w.add_box("b", (0.4, 0.4, 0.8), mass=60.0, pos=(3, -2, 1))
    post = w.anchor("post", pos=(1, 2, 1))
    sm = lobs.StateMachine("m")
    sm.var("n", 0.0)
    sm.var("dir", (0.0, 1.0, 0.0))
    sm.state(
        "guard", do=[act.steer(to=post, mode="arrive", speed=2.0, max_accel=5.0, stop_radius=0.2)]
    )
    sm.state("chase", on_enter=[act.emit("go"), act.set_var("n", props.behavior_var("ba", "n") + 1.0), act.enable_force("drag", False)],
             on_exit=[act.enable_force("drag", True)], do=[act.steer(to=ball, mode="intercept", speed=4.0, max_accel=8.0)])  # fmt: skip
    sm.state("wait", do=[act.steer(mode="hold"), act.force(ops.vec(0.0, 0.0, 0.0), max_force=1.0)])
    sm.transition("guard", "chase", when=rel.between(a, ball).distance() < 3.0, min_time=0.1)
    sm.transition("chase", "wait", when=ops.event("go").hold(0.5), priority=0)
    sm.transition(
        "chase", "guard", when=~(rel.between(a, ball).distance() < 4.5).hold(0.4), priority=2
    )
    sm.transition("wait", "guard", when=props.behavior_time_in_state("ba") > 0.6)
    w.add_behavior("ba", on=a, spec=sm, latency=0.15, period=0.05)
    sm2 = lobs.StateMachine("m2")
    sm2.state("flee", do=[act.steer(to=ball, mode="flee", speed=2.0, gain=3.0, plane=False)],
              on_enter=[act.enable_behavior("ba", True)])  # fmt: skip
    w.add_behavior("bb", on=b, spec=sm2, period=0.03)
    w.add_force("drag", on=ball, vector=ops.as_unit(props.velocity(ball) * -0.2, "newton"))
    w.add_force(
        "pull",
        on=b,
        toward=ball,
        magnitude=ops.as_unit(rel.between(b, ball).distance() * 0.5, "newton"),
        reaction=True,
    )
    w.add_adversary("gust", on=ball, budget=lobs.Budget(peak=1.5, impulse=0.4, active_time=0.8), vector=ops.vec(0, 1.5, 0),
                    when=(props.position(ball).x > -1.0) & (props.position(ball).x < 2.0))  # fmt: skip
    w.measure("state", props.behavior_state("ba"))
    w.measure("in_chase", props.behavior_in_state("ba", "chase"))
    w.measure("n", props.behavior_var("ba", "n"))
    w.measure("dirv", props.behavior_var("ba", "dir"))
    w.measure("go", ops.event("go"))
    w.measure("imp", props.force_impulse("gust"))
    w.measure("fa", props.force_applied("pull"))
    w.measure("gap", rel.between(a, ball).distance())
    return w


def test_machine_features_match():
    a, b = run_both(machine_world, 3.0)
    assert_same(a, b)
    assert a._beh.log == b._beh.log  # same transitions at the same times
    assert [(x.state_name, x.enter_t) for x in a._beh.items] == [
        (x.state_name, x.enter_t) for x in b._beh.items
    ]


def pendulum_world():
    w = lobs.World("p", dt=0.002, substeps=5)
    p = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = p.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0, collision_group=3)
    base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5), damping=0.05),
                   geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5, collision_group=3)  # fmt: skip
    w.spec.contact.disabled_group_pairs.append((3, 3))
    p.add_actuator("motor", "hinge", kind="effort", ctrl_range=(-3.0, 3.0))
    p.set_initial(q={"hinge": 0.4})
    sm = lobs.StateMachine("pd")
    sm.state("go", do=[act.actuate("pend/motor", ops.as_unit(-8.0 * props.joint_position("pend/hinge"), "newton*meter")
                                   + ops.as_unit(-1.5 * props.joint_velocity("pend/hinge"), "newton*meter"))])  # fmt: skip
    w.add_behavior("pd", on="pend", spec=sm)
    w.measure("theta", props.joint_position("pend/hinge"))
    w.measure("effort", props.joint_effort("pend/hinge"))
    return w


def test_owned_actuator_and_joint_state_match():
    a, b = run_both(pendulum_world, 2.0, controller=lambda s: {"pend/motor": 3.0})
    assert_same(a, b)
    assert np.allclose(a._action, b._action)
    assert abs(float(a["theta"])) < 0.05


def forces_world():
    w = lobs.World("f", dt=0.002, substeps=5, gravity=(0, 0, -9.81))
    w.add_box("floor", (10, 10, 0.2), fixed=True, pos=(0, 0, -0.1))
    a = w.add_box("a", (0.3, 0.2, 0.1), mass=2.0, pos=(0, 0, 0.5), quat=(0.92, 0.2, 0.3, 0.1))
    b = w.add_sphere("b", 0.1, mass=0.4, pos=(0.5, 0.1, 1.2))
    w.add_force(
        "tow", on=a, toward=b, magnitude=ops.as_unit(rel.between(a, b).distance() * 3.0, "newton")
    )
    w.add_force("axis", on=a, axis=(a, "x"), magnitude=2.0)
    w.add_force("dirf", on=b, direction=(1.0, 0.0, 0.5), magnitude=0.7, when=ops.time() > 0.2)
    w.measure("pa", props.position(a))
    w.measure("qa", props.orientation(a))
    return w


def test_toward_axis_direction_laws_and_user_forces_match():
    import os

    sims = []
    for mode in ("0", "1"):
        os.environ["LOBS_NATIVE_CORE"] = mode
        s = forces_world().simulate("mujoco", seed=1)
        sims.append(s)
    os.environ.pop("LOBS_NATIVE_CORE", None)
    for s in sims:
        for k in range(150):
            if k in (10, 60):
                s.apply_force("a", (3.0, 0.0, 5.0), ticks=4)
            s.step()
    assert sims[1]._core is not None and sims[0]._core is None
    assert_same(*sims)


def test_script_behaviors_stay_on_the_python_runtime():
    @lobs.behavior("core_test_script")
    def _(ctx):
        ctx.force("ball", (0.1, 0, 0))

    w = lobs.World("s", gravity=(0, 0, 0))
    w.add_sphere("ball", 0.1, mass=1.0, pos=(0, 0, 1))
    w.add_behavior("s", script="core_test_script")
    s = w.simulate("mujoco")
    s.run(ticks=20)
    assert s._core is None and float(s.state().body_pos[0][0]) > 0
