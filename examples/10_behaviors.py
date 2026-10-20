"""Behaviors: a defender that decides, plus a budgeted gust that pushes the ball.

A force law is a stateless function of the state. A behavior owns state: this defender guards a post,
chases the ball when it comes close (reaction time 0.15 s, decisions every 0.05 s), and returns to its post
when the ball is gone. The gust is an adversarial force that only acts while the ball is in a region and has
a hard budget, so the pressure is bounded and the run reproducible.
"""

import lobs
from lobs import act, ops, props, rel

w = lobs.World("defence", dt=0.002, substeps=5, gravity=(0, 0, 0))
ball = w.add_sphere("ball", 0.12, mass=0.6, pos=(-4, 0, 1))
defender = w.add_box("defender", (0.5, 0.5, 1.0), mass=70.0, pos=(1, 3, 1))
post = w.anchor("post", pos=(1, 3, 1))

gap = rel.between(defender, ball).distance()
sm = lobs.StateMachine("defend")
sm.state("guard", do=[act.steer(to=post, mode="arrive", speed=2.0, max_accel=5.0)])
sm.state("chase", on_enter=[act.emit("engaged")], do=[act.steer(to=ball, mode="intercept", speed=4.0, max_accel=8.0)])
sm.transition("guard", "chase", when=gap < 3.5, min_time=0.1)
sm.transition("chase", "guard", when=~(gap < 4.5).hold(0.5))
w.add_behavior("defender", on=defender, spec=sm, latency=0.15, period=0.05)

w.add_adversary(
    "gust", on=ball, budget=lobs.Budget(peak=1.5, impulse=0.6),
    vector=ops.vec(0, 1.5, 0), when=(props.position(ball).x > -1.0) & (props.position(ball).x < 2.0),
)  # fmt: skip

w.measure("gap", gap)
w.measure("state", props.behavior_state("defender"))
w.measure("engaged", ops.event("engaged"))
print(w.validate())

sim = w.simulate("mujoco", seed=0)
sim.reset(initial_state={"ball": lobs.Initial(pos=(-4, 0, 1), linvel=(3.0, 0, 0))})
last = None
for _ in range(int(3.0 / sim.tick_dt)):
    sim.step()
    names = ["guard", "chase"]
    now = names[int(sim["state"])]
    if now != last:
        print(f"t={sim.time:5.2f}s  defender -> {now:6}  (gap {sim['gap']:.2f} m)")
        last = now

stats = sim.force_stats["gust"]
print(f"gust delivered {stats['impulse']:.3f} N s of its 0.600 budget, peak {stats['peak']:.2f} N")
assert abs(stats["impulse"] - 0.6) < 1e-6 and last == "chase"
