import numpy as np
import pytest

import lobs
from lobs import ops, props, rel
from lobs.errors import LobsError, WorldValidationError

BACKENDS = lobs.backends.available()


def two_spheres(m1=2.0, m2=3.0, d0=2.0, dt=0.001):
    w = lobs.World("pair", dt=dt, substeps=5, gravity=(0, 0, 0))
    a = w.add_sphere("a", 0.1, mass=m1, pos=(-d0 / 2, 0, 1), collision_group=1)
    b = w.add_sphere("b", 0.1, mass=m2, pos=(d0 / 2, 0, 1), collision_group=1)
    w.spec.contact.disabled_group_pairs.append((1, 1))
    return w, a, b


@pytest.mark.parametrize("be", BACKENDS)
def test_constant_direction_force_accelerates_by_f_over_m(be):
    w = lobs.World(dt=0.002, substeps=5, gravity=(0, 0, 0))
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=2.0, pos=(0, 0, 1))
    w.add_force("wind", on=box, magnitude=4.0, direction=(1, 0, 0))
    w.measure("vx", props.velocity(box).x)
    sim = w.simulate(be)
    sim.run(seconds=1.0)
    assert sim["vx"] == pytest.approx(2.0, rel=2e-3)  # a = F/m = 2 m/s^2


def inverse_cube_reference(m1, m2, d0, k, t_end, n=20000):
    """r'' = -k / (mu r^3), integrated with RK4 (the independent answer)."""
    mu = m1 * m2 / (m1 + m2)
    f = lambda y: np.array([y[1], -k / (mu * y[0] ** 3)])  # noqa: E731
    y, h = np.array([d0, 0.0]), t_end / n
    for _ in range(n):
        k1 = f(y)
        k2 = f(y + h / 2 * k1)
        k3 = f(y + h / 2 * k2)
        k4 = f(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[0]


@pytest.mark.parametrize("be", BACKENDS)
def test_inverse_cube_attraction_with_reaction_matches_two_body_ode(be):
    m1, m2, d0, k = 2.0, 3.0, 2.0, 8.0
    w, a, b = two_spheres(m1, m2, d0)
    dist = rel.between(a, b).distance()
    w.add_force("grav", on=a, toward=b, magnitude=k / dist**3)  # reaction on b by default
    w.measure("r", dist)
    w.measure("p", props.momentum(a) + props.momentum(b))
    w.measure("com_x", (props.position(a).x * m1 + props.position(b).x * m2) / (m1 + m2))
    sim = w.simulate(be)
    sim.run(seconds=0.6)
    expected = inverse_cube_reference(m1, m2, d0, k, sim.time)
    tol = 2e-3 if be == "mujoco" else 1e-2
    assert sim["r"] == pytest.approx(expected, rel=tol)
    assert sim["r"] < d0 - 0.05  # they really fell toward each other
    assert np.abs(sim["p"]).max() < 1e-6 * (
        1 if be == "mujoco" else 1e4
    )  # Newton's third law: no net momentum
    com0 = (-d0 / 2 * m1 + d0 / 2 * m2) / (
        m1 + m2
    )  # the centre of mass stays exactly where it started
    assert sim["com_x"] == pytest.approx(com0, abs=1e-4 if be == "mujoco" else 1e-2)


@pytest.mark.parametrize("be", BACKENDS)
def test_reaction_flag_and_repulsion_sign(be):
    w, a, b = two_spheres()
    w.add_force("push", on=a, toward=b, magnitude=-5.0, reaction=False)  # negative = away from b
    w.measure("va", props.velocity(a).x)
    w.measure("vb", props.velocity(b).x)
    sim = w.simulate(be)
    sim.run(seconds=0.5)
    assert sim["va"] == pytest.approx(
        -5.0 / 2.0 * 0.5, rel=2e-3
    )  # a (at -x) is pushed further to -x
    assert sim["vb"] == pytest.approx(0.0, abs=1e-6)  # no reaction


@pytest.mark.parametrize("be", BACKENDS)
def test_spring_along_an_anchor_axis_oscillates_at_the_right_frequency(be):
    k, m = 50.0, 2.0
    w = lobs.World(dt=0.001, substeps=2, gravity=(0, 0, 0))
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=m, pos=(0, 0, 1.3))
    rest = w.anchor(
        "rest", pos=(0, 0, 1.0)
    )  # world anchor: a fixed frame whose z axis the force follows
    h = props.height(box) - 1.0
    w.add_force("spring", on=box, magnitude=-k * h, axis=(rest, "z"))
    w.measure("z", props.height(box))
    sim = w.simulate(be)
    sim.run(seconds=3.0)
    z, t = sim.history["z"], sim.history.time
    up = np.where((z[:-1] < 1.0) & (z[1:] >= 1.0))[0]
    period = np.mean(np.diff(t[up]))
    assert period == pytest.approx(2 * np.pi * np.sqrt(m / k), rel=0.01)
    assert (
        z.max() < 1.3 + 0.02 and z.min() > 0.7 - 0.02
    )  # energy conserved by the symplectic-ish scheme


@pytest.mark.parametrize("be", BACKENDS)
def test_force_on_an_anchor_acts_at_that_point_and_torques_the_body(be):
    w = lobs.World(dt=0.001, substeps=5, gravity=(0, 0, 0))
    box = w.add_box("box", (1, 1, 1), mass=1.0, pos=(0, 0, 1))
    tip = box.anchor("tip", pos=(0.5, 0, 0))
    w.add_force("kick", on=tip, magnitude=10.0, direction=(0, 1, 0))
    w.measure("L", props.angular_momentum(box))
    w.measure("p", props.momentum(box))
    sim = w.simulate(be)
    sim.run(seconds=0.1)
    assert sim["p"][1] == pytest.approx(10.0 * 0.1, rel=2e-3)
    assert sim["L"][2] == pytest.approx(0.5 * 10.0 * 0.1, rel=0.02)  # tau = r x F = (0,0,5)


@pytest.mark.parametrize("be", BACKENDS)
def test_force_on_an_articulated_actor_is_shared_by_mass(be):
    w = lobs.World(dt=0.002, substeps=5, gravity=(0, 0, 0))
    a = w.add_actor("arm", root="free", pos=(0, 0, 2))
    base = a.add_body("base", geoms=[lobs.Box(size=(0.2, 0.2, 0.2))], mass=3.0, collision_group=2)
    base.add_child("link", pos=(0.5, 0, 0), joint=lobs.Fixed("weld"), geoms=[lobs.Box(size=(0.2, 0.2, 0.2))], mass=1.0, collision_group=2)  # fmt: skip
    w.spec.contact.disabled_group_pairs.append((2, 2))
    w.add_force("push", on=a, magnitude=8.0, direction=(0, 0, 1))
    w.measure("vz", props.velocity(a).z)
    w.measure("spin", props.spin(a))
    sim = w.simulate(be)
    sim.run(seconds=0.5)
    assert sim["vz"] == pytest.approx(8.0 / 4.0 * 0.5, rel=3e-3)  # a = F / (m1 + m2)
    assert sim["spin"] < 1e-3  # shared by mass: pure translation, no torque about the actor COM


def test_force_laws_validate_roundtrip_and_include(tmp_path):
    w, a, b = two_spheres()
    w.add_force("g", on=a, toward=b, magnitude=1.0 / rel.between(a, b).distance() ** 2)
    assert w.validate().ok
    w.save(tmp_path / "f.yaml")
    w2 = lobs.load_world(tmp_path / "f.yaml")
    assert w2.spec.forces == w.spec.forces and w2.validate().ok
    host = lobs.World("host", gravity=(0, 0, 0))
    host.include(w2, prefix="p")
    assert (
        host.spec.forces[0].name == "p.g"
        and host.spec.forces[0].on == "p.a"
        and host.spec.forces[0].toward == "p.b"
    )
    sim = host.simulate("mujoco")
    sim.run(seconds=0.1)
    assert sim.reg.resolve("p.a")
    assert not any(n.startswith("__") for n in sim.measurements())  # hidden magnitudes stay hidden


def test_force_law_errors_are_clear():
    w, a, b = two_spheres()
    w.add_force("bad1", on=a, magnitude=1.0)  # no direction at all
    assert any("exactly one of" in i.message for i in w.validate().errors)
    w2, a, b = two_spheres()
    w2.add_force("bad2", on=a, toward=b, direction=(1, 0, 0), magnitude=1.0)
    assert not w2.validate().ok
    w3, a, b = two_spheres()
    w3.add_force("vec", on=a, toward=b, magnitude=props.position(a))  # not a scalar
    assert any("scalar" in i.message for i in w3.validate().errors)
    w4, a, b = two_spheres()
    ground = w4.anchor("fixed_pt", pos=(0, 0, 0))
    w4.add_force("onworld", on=ground, toward=a, magnitude=1.0)
    assert any("world anchor" in i.message for i in w4.validate().errors)
    with pytest.raises(WorldValidationError):
        w4.simulate("mujoco")
    w5, a, b = two_spheres()
    w5.add_force(
        "nan",
        on=a,
        toward=b,
        magnitude=ops.where(props.height(a) > 0, props.height(a) / 0.0 * 0.0, 1.0),
    )
    sim = w5.simulate("mujoco")
    with pytest.raises(LobsError, match="non-finite"):
        sim.step()
    with pytest.raises(LobsError, match="already exists"):
        w5.add_force("nan", on=a, toward=b, magnitude=1.0)
