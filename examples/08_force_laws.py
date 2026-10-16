"""Strange physics: custom force laws. A force is a measure-valued magnitude along a direction.

Two bodies attract with an inverse-CUBE law (not our universe's inverse-square), a gust pushes one for a
fraction of a second, and a spring acts along an anchor's axis. Compare against an independent ODE.
"""

import numpy as np

import lobs
from lobs import props, rel

m1, m2, d0, k = 2.0, 3.0, 2.0, 8.0
world = lobs.World("strange", dt=0.001, substeps=5, gravity=(0, 0, 0))
a = world.add_sphere("a", 0.1, mass=m1, pos=(-d0 / 2, 0, 1), collision_group=1)
b = world.add_sphere("b", 0.1, mass=m2, pos=(d0 / 2, 0, 1), collision_group=1)
world.spec.contact.disabled_group_pairs.append((1, 1))

dist = rel.between(a, b).distance()
world.add_force("inverse_cube", on=a, toward=b, magnitude=k / dist**3)  # reaction on b by default
world.measure("r", dist)
world.measure("momentum", props.momentum(a) + props.momentum(b))

c = world.add_box("c", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 5, 1))
world.add_disturbance("gust", on=c, force=(3.0, 0, 0), start=0.1, duration=0.2)  # 3 N for 0.2 s
world.measure("c_vx", props.velocity(c).x)

sim = world.simulate("mujoco")
sim.run(seconds=0.6)

# independent answer: relative motion r'' = -k / (mu r^3), mu = reduced mass
mu, (r, rd), h = m1 * m2 / (m1 + m2), (d0, 0.0), 0.6 / 20000
for _ in range(20000):
    f = lambda r, rd: (rd, -k / (mu * r**3))  # noqa: E731
    k1 = f(r, rd)
    k2 = f(r + h / 2 * k1[0], rd + h / 2 * k1[1])
    k3 = f(r + h / 2 * k2[0], rd + h / 2 * k2[1])
    k4 = f(r + h * k3[0], rd + h * k3[1])
    r, rd = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), rd + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
print(f"separation after 0.6 s: simulated {sim['r']:.4f} m, ODE {r:.4f} m")
print("net momentum (Newton's third law):", np.abs(sim["momentum"]).max().round(9))
print(f"gust impulse: v = {sim['c_vx']:.3f} m/s (expected 3 N * 0.2 s / 1 kg = 0.6)")
