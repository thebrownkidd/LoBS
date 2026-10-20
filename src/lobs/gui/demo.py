"""The built-in demo world: a ball bounces around a room while a defender with a state machine chases it,
and a budgeted gust pushes the ball. Everything moves as soon as you press Play."""

from __future__ import annotations

import lobs
from lobs import act, ops, props, rel


def build_demo() -> lobs.World:
    w = lobs.World("defence-demo", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_box("floor", (14, 9, 0.2), fixed=True, pos=(1, 0, -0.1))
    w.add_box("wall", (0.3, 9, 2.0), fixed=True, pos=(7.0, 0, 1.0), restitution=0.9)
    w.add_box("back_wall", (0.3, 9, 2.0), fixed=True, pos=(-5.5, 0, 1.0), restitution=0.9)
    ball = w.add_sphere(
        "ball",
        0.12,
        mass=0.6,
        pos=(-4, 0, 1),
        restitution=0.9,
        linvel=(3.0, 0.3, 0),
        rgba=(0.95, 0.5, 0.15, 1.0),
    )
    defender = w.add_box(
        "defender", (0.5, 0.5, 1.0), mass=70.0, pos=(1, 3, 1), rgba=(0.3, 0.55, 0.9, 1.0)
    )
    post = w.anchor("post", pos=(1, 3, 1))

    gap = rel.between(defender, ball).distance()
    sm = lobs.StateMachine("defend")
    sm.state("guard", do=[act.steer(to=post, mode="arrive", speed=2.0, max_accel=5.0)])
    sm.state(
        "chase",
        on_enter=[act.emit("engaged")],
        do=[act.steer(to=ball, mode="intercept", speed=4.0, max_accel=8.0)],
    )
    sm.transition("guard", "chase", when=gap < 3.5, min_time=0.1)
    sm.transition("chase", "guard", when=~(gap < 4.5).hold(0.5))
    w.add_behavior("defender", on=defender, spec=sm, latency=0.15, period=0.05)

    w.add_adversary("gust", on=ball, budget=lobs.Budget(peak=1.5, impulse=0.6), vector=ops.vec(0, 1.5, 0),
                    when=(props.position(ball).x > -1.0) & (props.position(ball).x < 2.0))  # fmt: skip
    w.measure("gap", gap)
    w.measure("ball_speed", props.speed(ball))
    w.measure("engaged", ops.event("engaged"))
    return w


def static_hint(world) -> str | None:
    """A hint when pressing Play would show nothing: no motion, no gravity, no forces, no behaviors."""
    s = world.spec
    moving = any(
        any(abs(x) > 1e-12 for x in (*a.initial.linvel, *a.initial.angvel)) or a.initial.qd
        for a in s.actors
    )
    gravity = any(abs(g) > 1e-12 for g in s.gravity) and any(a.root == "free" for a in s.actors)
    if (
        moving
        or gravity
        or s.forces
        or s.behaviors
        or s.anchors
        and any(getattr(x.rule, "type", "rigid") == "path" for x in s.anchors)
    ):
        return None
    return "Nothing moves in this world: no gravity, forces, behaviors or initial velocities. Give something a velocity in the Inspector (initial.linvel) or add a force."
