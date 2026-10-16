"""Record once, then add anchors and measures afterwards and evaluate them without re-simulating."""

from pathlib import Path

import lobs
from lobs import props, rel

OUT = Path("runs")

world = lobs.World("replay", dt=0.002, substeps=5)
world.add_plane("ground")
ball = world.add_sphere("ball", radius=0.1, mass=1.0, pos=(0, 0, 2), linvel=(1.0, 0, 3.0))
world.measure("height", props.height(ball))
sim = world.simulate("mujoco")
sim.run(seconds=1.5)
sim.export(OUT / "ball.npz")

# --- later: a different analysis of the same recorded run
traj = lobs.Trajectory.load(OUT / "ball.npz")
target = traj.world.anchor("target", pos=(2, 0, 0))  # an anchor that did not exist when we simulated
distance = traj.evaluate(rel.between(traj.world.actor("ball"), target).distance())
speed = traj.evaluate(props.speed(traj.world.actor("ball")))
print(f"closest approach to target: {distance.min():.3f} m at t={traj.t[distance.argmin()]:.2f} s")
print(f"peak speed: {speed.max():.2f} m/s")
