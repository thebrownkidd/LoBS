"""An articulated pendulum with a control loop; properties (props) and relative measures (rel)."""

import numpy as np

import lobs
from lobs import props, rel

world = lobs.World("pendulum", dt=0.002, substeps=5)
pend = world.add_actor("pend", root="fixed", pos=(0, 0, 2))
base = pend.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0)
rod = base.add_child(
    "rod",
    pos=(0, 0, -0.5),
    joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5), damping=0.02),
    geoms=[lobs.Capsule(radius=0.02, length=1.0)],
    mass=0.5,
    collision_group=1,
)
world.spec.contact.disabled_group_pairs.append((1, 1))
pend.add_actuator("motor", joint="hinge", kind="effort", ctrl_range=(-5, 5))
tip = rod.anchor("tip", pos=(0, 0, -0.5))
pivot = base.anchor("pivot", pos=(0, 0, 0))

theta = world.measure("theta", props.joint_position(pend.joint("hinge")))
world.measure("tip_height", props.height(tip))
world.measure("tip_speed", props.speed(tip))
world.measure("angle_from_down", rel.between(pivot, tip).angle("polar", axis="-z", frame="world"))
world.measure("energy", props.mechanical_energy(rod))
world.measure("swing", theta.running_max())

pend.set_initial(q={"hinge": 1.0})
with world.simulate("mujoco") as sim:
    while sim.time < 5.0:
        sim.step(action={"motor": -2.0 * sim["theta"]})  # a simple PD-ish controller
    print(f"theta {sim['theta']:+.3f} rad (max swing {sim['swing']:.3f}), tip height {sim['tip_height']:.3f} m")
    print("angle from straight down:", np.degrees(sim["angle_from_down"]).round(2), "deg")
