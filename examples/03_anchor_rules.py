"""Anchors that move on their own: orbit, sine path, keyframes, and a 'level' (translate-only) frame."""

import numpy as np

import lobs
from lobs import props, rel

world = lobs.World("anchors", dt=0.01)
world.add_plane("ground")
centre = world.anchor("centre", pos=(0, 0, 1))
orbit = world.anchor("orbit", pos=(0, 0, 1), rule=lobs.Path.circle(radius=2.0, period=4.0))
wobble = world.anchor("wobble", pos=(3, 0, 1), rule=lobs.Path.sine(amplitude=0.5, frequency=0.5))
cart = world.anchor("cart", pos=(-3, 0, 0.5), rule=lobs.Path.keyframed([(0, (0, 0, 0)), (2, (2, 0, 0)), (4, (2, 2, 0))]))

azimuth = world.measure("azimuth", rel.between(centre, orbit).angle("azimuth", axis="z", frame="world"))
world.measure("azimuth_rate", azimuth.rate())  # exact, and no spike where the angle wraps at +-pi
world.measure("orbit_speed", props.speed(orbit))
world.measure("wobble_dist", rel.between(centre, wobble).distance())
world.measure("cart_x", props.position(cart).x)

sim = world.simulate("mujoco")
sim.run(seconds=4.0)
print("azimuth rate (rad/s):", float(np.mean(sim.history["azimuth_rate"])), "expected", 2 * np.pi / 4.0)
print("orbit speed (m/s):", float(np.mean(sim.history["orbit_speed"])), "expected", 2 * np.pi * 2.0 / 4.0)
print("cart x at t=4:", sim["cart_x"])
