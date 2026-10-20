"""Golden scenarios: broad, deterministic runs whose full measure histories pin the numerical behaviour.

Used by tools/make_golden.py (record) and tests/test_golden.py (compare), so speed-ups can be proven to keep
results identical."""

from __future__ import annotations

import numpy as np

import lobs
from lobs import act, ops, props, rel
from lobs.gui.demo import build_demo


def demo():
    return build_demo(), dict(seconds=3.0)


def zoo():
    w = lobs.World("zoo", dt=0.002, substeps=5)
    w.add_box("floor", (10, 10, 0.2), fixed=True, pos=(0, 0, -0.1), restitution=0.3)
    crate = w.add_box(
        "crate",
        (0.3, 0.2, 0.1),
        mass=2.0,
        pos=(0, 0, 0.6),
        quat=(0.92, 0.2, 0.3, 0.1),
        restitution=0.3,
    )
    ball = w.add_sphere(
        "ball", 0.1, mass=0.4, pos=(0.6, 0.1, 1.2), linvel=(-0.5, 0.0, 1.0), restitution=0.5
    )
    pend = w.add_actor("pend", root="fixed", pos=(2, 0, 2))
    base = pend.add_body(
        "base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0, collision_group=3
    )
    base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5), damping=0.05),
                   geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5, collision_group=3)  # fmt: skip
    w.spec.contact.disabled_group_pairs.append((3, 3))
    pend.add_actuator("motor", "hinge", kind="position", kp=20.0, kd=2.0, ctrl_range=(-1.5, 1.5))
    pend.set_initial(q={"hinge": 0.3})
    tip = pend.anchor("tip", pos=(0, 0, -0.5), body="rod")
    orb = crate.anchor("orb", pos=(0.1, 0, 0.05), rule=lobs.Path.circle(0.05, 0.5))
    ms = {
        "pos": props.position(ball),
        "vel": props.velocity(ball),
        "speed": props.speed(ball),
        "ke": props.kinetic_energy(crate),
        "pe": props.potential_energy(ball),
        "me": props.mechanical_energy(ball),
        "q": props.orientation(crate),
        "tilt": props.tilt(crate),
        "heading": props.heading(crate),
        "w": props.angular_velocity(crate),
        "mom": props.momentum(ball),
        "L": props.angular_momentum(crate),
        "jq": props.joint_position(pend.joint("hinge")),
        "jqd": props.joint_velocity(pend.joint("hinge")),
        "tip_pos": props.position(tip),
        "tip_vel": props.velocity(tip),
        "orb_pos": props.position(orb),
        "orb_vel": props.velocity(orb),
        "d": rel.between(ball, crate).distance(),
        "d_tip": rel.between(tip, ball).distance(),
        "ang_polar": rel.between(crate, ball).angle("polar"),
        "ang_elev": rel.between(crate, ball).angle("elevation"),
        "ang_az": rel.between(crate, ball).angle("azimuth"),
        "touch": props.in_contact(ball, crate),
        "touch_floor": props.in_contact(crate, "world/floor") if False else props.in_contact(crate),
        "z_rate": props.position(ball).z.rate(),
        "d_rate": rel.between(ball, crate).distance().rate(),
        "z_int": props.position(ball).z.integral(),
        "z_max": props.position(ball).z.running_max(),
        "z_win": props.position(ball).z.window(0.2).mean(),
        "z_der": props.position(ball).z.derivative(),
        "low": props.position(ball).z < 0.5,
        "rise": (props.position(ball).z < 0.5).rising(),
        "latch": ops.latch(
            (props.position(ball).z < 0.4).rising(), (props.position(ball).z > 1.0).rising()
        ),
        "held": (props.position(ball).z < 0.3).hold(0.3),
        "cool": (props.speed(ball) > 1.0).cooldown(0.2),
        "dot": ops.dot(props.velocity(ball), props.velocity(crate)),
        "cr": ops.cross(props.velocity(ball), props.position(ball)),
        "nrm": ops.norm(props.velocity(ball)),
        "unit": ops.unit(props.velocity(ball)),
        "clip": ops.clip_norm(props.velocity(ball), 0.5),
        "wh": ops.where(props.position(ball).z > 0.5, 1.0, -1.0),
        "mix": ops.sin(ops.as_unit(props.position(ball).x, "radian")) * props.speed(ball)
        + ops.cos(ops.as_unit(props.position(ball).y, "radian")) * props.speed(ball),
        "tsince": (props.position(ball).z < 0.5).time_since(),
        "dur": (props.position(ball).z < 0.5).duration_true(),
    }
    for k, m in ms.items():
        w.measure(k, m)
    return w, dict(seconds=2.0, controller=lambda s: {"pend/motor": 1.0 * np.sin(3.0 * s.time)})


def behaviors():
    w = lobs.World("beh", dt=0.002, substeps=5, gravity=(0, 0, 0))
    ball = w.add_sphere("ball", 0.1, mass=0.5, pos=(-3, 0, 1), linvel=(2.5, 0.2, 0))
    a = w.add_box("a", (0.4, 0.4, 0.8), mass=60.0, pos=(1, 2, 1))
    b = w.add_box("b", (0.4, 0.4, 0.8), mass=60.0, pos=(3, -2, 1))
    post = w.anchor("post", pos=(1, 2, 1))
    sm = lobs.StateMachine("m")
    sm.var("n", 0.0)
    sm.state("guard", do=[act.steer(to=post, mode="arrive", speed=2.0, max_accel=5.0)])
    sm.state("chase", on_enter=[act.emit("go"), act.set_var("n", props.behavior_var("a_beh", "n") + 1.0)],
             do=[act.steer(to=ball, mode="intercept", speed=4.0, max_accel=8.0)])  # fmt: skip
    sm.transition("guard", "chase", when=rel.between(a, ball).distance() < 3.0, min_time=0.1)
    sm.transition(
        "chase", "guard", when=~(rel.between(a, ball).distance() < 4.5).hold(0.4), priority=1
    )
    w.add_behavior("a_beh", on=a, spec=sm, latency=0.15, period=0.05)
    sm2 = lobs.StateMachine("m2")
    sm2.state(
        "flee", do=[act.steer(to=ball, mode="flee", speed=2.0), act.force(ops.vec(0.0, 0.0, 0.0))]
    )
    w.add_behavior("b_beh", on=b, spec=sm2)
    w.add_adversary("gust", on=ball, budget=lobs.Budget(peak=1.5, impulse=0.6), vector=ops.vec(0, 1.5, 0),
                    when=(props.position(ball).x > -1.0) & (props.position(ball).x < 2.0))  # fmt: skip
    w.add_force("drag", on=ball, vector=ops.as_unit(props.velocity(ball) * -0.1, "newton"))
    w.add_force(
        "pull",
        on=b,
        toward=ball,
        magnitude=ops.as_unit(rel.between(b, ball).distance() * 0.5, "newton"),
    )
    for k, m in {
        "gap": rel.between(a, ball).distance(),
        "state": props.behavior_state("a_beh"),
        "n": props.behavior_var("a_beh", "n"),
        "tis": props.behavior_time_in_state("a_beh"),
        "go": ops.event("go"),
        "imp": props.force_impulse("gust"),
        "act": props.force_active("gust"),
        "fa": props.force_applied("gust"),
        "pa": props.position(a),
        "pb": props.position(b),
        "pball": props.position(ball),
        "coin": ops.chance(0.3),
        "u": ops.uniform(0.0, 1.0),
        "g": ops.normal(0.0, 1.0),
        "samp": ops.uniform(0.0, 1.0).sample_on(rel.between(a, ball).distance() < 3.0),
    }.items():
        w.measure(k, m)
    return w, dict(seconds=3.0, seed=7)


SCENARIOS = {"demo": demo, "zoo": zoo, "behaviors": behaviors}


def run(name: str, backend: str = "mujoco") -> dict:
    world, kw = SCENARIOS[name]()
    seed = kw.pop("seed", 0)
    sim = world.simulate(backend, seed=seed, record=True)
    sim.run(**kw)
    out = {"t": sim.history.time}
    for n in sim.history.names():
        out[f"m/{n}"] = np.asarray(sim.history[n], dtype=float)
    fr = sim.state()
    out["final/pos"], out["final/quat"] = fr.body_pos, fr.body_quat
    out["final/lin"], out["final/ang"] = fr.body_linvel, fr.body_angvel
    for k, v in sim.force_stats.items():
        out[f"force/{k}/impulse"] = np.asarray(v["impulse"])
        out[f"force/{k}/peak"] = np.asarray(v["peak"])
    sim.close()
    return out
