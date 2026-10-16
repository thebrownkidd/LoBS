"""Comparison mode: the same pendulum on MuJoCo and Chrono, side by side."""

import lobs
from lobs import props

world = lobs.World("pendulum", dt=0.002)
pend = world.add_actor("pend", root="fixed", pos=(0, 0, 2))
base = pend.add_body("base", geoms=[lobs.Box(size=(0.05, 0.05, 0.05))], mass=1.0)
base.add_child(
    "rod",
    pos=(0, 0, -0.5),
    joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5)),
    geoms=[lobs.Capsule(radius=0.02, length=1.0)],
    mass=0.5,
    collision_group=7,
)
world.spec.contact.disabled_group_pairs.append((7, 7))
pend.set_initial(q={"hinge": 0.6})
world.measure("theta", props.joint_position(pend.joint("hinge")))
world.measure("tip_speed", props.speed(pend.body("rod")))
world.measure("energy", props.mechanical_energy(pend.body("rod")))

result = lobs.compare(world, ["mujoco", "chrono"], seconds=4.0)
print(result.table())
