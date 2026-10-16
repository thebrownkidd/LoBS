"""A Gymnasium environment: you choose the observations, the reward and when episodes end.

The simulator prescribes no task. Here: swing a pendulum up with a bounded motor (a classic), written
entirely in terms of measures.
"""

import numpy as np

import lobs
from lobs import props

world = lobs.World("swingup", dt=0.002, substeps=5)  # 10 ms control steps
pend = world.add_actor("pend", root="fixed", pos=(0, 0, 2))
base = pend.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0, collision_group=1)
base.add_child(
    "rod",
    pos=(0, 0, -0.5),
    joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5), damping=0.02),
    geoms=[lobs.Capsule(radius=0.02, length=1.0)],
    mass=0.5,
    collision_group=1,
)
world.spec.contact.disabled_group_pairs.append((1, 1))
pend.add_actuator("motor", "hinge", kind="effort", ctrl_range=(-2.0, 2.0))
theta = world.measure("theta", props.joint_position(pend.joint("hinge")))
world.measure("omega", props.joint_velocity(pend.joint("hinge")))
world.measure("tip_height", props.height(pend.body("rod")))

env = lobs.GymEnv(
    world,
    "mujoco",
    observe=["theta", "omega"],  # which measures the agent sees
    reward=lambda sim: sim["tip_height"],  # the task is yours: here, tip height
    terminated=lambda sim: False,
    max_seconds=5.0,
    normalize_actions=True,  # actions in [-1, 1] -> the motor's control range
    randomize={"pend": {"q": {"hinge": lobs.Uniform(-0.3, 0.3)}}},  # a different start each reset
)

obs, info = env.reset(seed=0)
total, steps = 0.0, 0
while True:
    action = np.array([np.sign(obs[1]) or 1.0])  # push along the current motion (energy pumping)
    obs, reward, terminated, truncated, info = env.step(action)
    total += reward
    steps += 1
    if terminated or truncated:
        break
print(f"{steps} steps, return {total:.1f}, final theta {info['measurements']['theta']:+.2f} rad")
env.close()
