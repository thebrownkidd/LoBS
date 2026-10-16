"""See what happened: offscreen images (any backend) and a Rerun recording."""

from pathlib import Path

import lobs
from lobs import props

OUT = Path("runs")

world = lobs.World("tower", dt=0.002, substeps=5)
world.add_plane("ground")
for i in range(4):  # a little tower that gets hit by a ball
    world.add_box(f"block{i}", (0.4, 0.4, 0.4), mass=1.0, pos=(0, 0, 0.2 + 0.4 * i))
ball = world.add_sphere("ball", 0.15, mass=2.0, pos=(-2.5, 0, 0.45), linvel=(6.0, 0, 0))
world.measure("ball_speed", props.speed(ball))

# the same code renders MuJoCo and Chrono runs (a MuJoCo "visual twin" is posed from LoBS state)
for backend in lobs.backends.available():
    sim = world.simulate(backend)
    sim.run(seconds=0.5)
    image = sim.render(640, 360, camera={"azimuth": 150, "elevation": -20, "distance": 7.0, "lookat": (0.3, 0, 0.8)})
    lobs.viz.write_png(OUT / f"tower_{backend}.png", image)
    print(f"{backend}: wrote runs/tower_{backend}.png {image.shape}")

# a Rerun recording: geometry once, body transforms / anchors / measures per tick (open with `rerun runs/tower.rrd`)
sim = world.simulate("mujoco")
sim.run(seconds=2.0)
lobs.viz.log_rerun(sim, "tower", save=OUT / "tower.rrd", every=5)
print("wrote runs/tower.rrd")
