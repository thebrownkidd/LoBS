"""Drop a box on the ground, measure it, export the run, save the world."""

from pathlib import Path

import lobs
from lobs import props, rel

OUT = Path("runs")

world = lobs.World("drop", dt=0.002, substeps=5)  # one tick = 10 ms
world.add_plane("ground", friction=0.8)
box = world.add_box("box", size=(0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
top = box.anchor_on_face("top")  # moves with the box
goal = world.anchor("goal", pos=(1, 0, 0.1))  # fixed in the world

world.measure("height", props.height(box))
gap = world.measure("gap", rel.between(top, goal).distance())
world.measure("closing", gap.rate())  # exact, from engine velocities

sim = world.simulate("mujoco", seed=0)
sim.run(seconds=2.0)
print(f"final height {sim['height']:.4f} m, gap {sim['gap']:.4f} m, closing speed {sim['closing']:.4f} m/s")
sim.export(OUT / "drop.npz")
world.save(OUT / "drop.yaml")
