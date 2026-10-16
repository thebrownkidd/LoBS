import json

import numpy as np
import pytest

import lobs
from lobs import props
from lobs.cli import main

from .fixtures.make_fixtures import make_gltf_room

BACKENDS = lobs.backends.available()


@pytest.mark.parametrize("be", BACKENDS)
def test_contact_vector_measures_on_a_resting_box(be):
    w = lobs.World(dt=0.002, substeps=2)
    ground = w.add_plane("ground")
    box = w.add_box("box", (0.4, 0.4, 0.4), mass=2.0, pos=(1.0, 0.5, 0.21))
    w.measure("n_box", props.contact_normal(box))
    w.measure("n_ground", props.contact_normal(ground))
    w.measure("f_box", props.contact_force_vector(box))
    w.measure("f_ground", props.contact_force_vector(ground))
    w.measure("pt", props.contact_point(box))
    w.measure("depth", props.contact_depth(box))
    w.measure("depth_none", props.contact_depth(box, with_=ground))
    sim = w.simulate(be)
    sim.run(seconds=1.5)
    weight = 2.0 * 9.81
    assert np.allclose(sim["n_box"], (0, 0, 1), atol=1e-3)  # the ground pushes the box up
    assert np.allclose(
        sim["n_ground"], (0, 0, -1), atol=1e-3
    )  # ... and the box pushes the ground down
    assert np.allclose(sim["f_box"], (0, 0, weight), rtol=0.03, atol=0.3)
    assert np.allclose(sim["f_ground"], (0, 0, -weight), rtol=0.03, atol=0.3)  # Newton's third law
    pt = sim["pt"]
    assert (
        abs(pt[2]) < 5e-3 and abs(pt[0] - 1.0) < 0.21 and abs(pt[1] - 0.5) < 0.21
    )  # on the box's footprint
    assert 0.0 <= sim["depth"] < 5e-3 and sim["depth_none"] == sim["depth"]
    # a body that touches nothing: no direction, no point, zero depth
    free = lobs.World(dt=0.002, gravity=(0, 0, 0))
    b = free.add_box("b", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 3))
    free.measure("n", props.contact_normal(b))
    free.measure("p", props.contact_point(b))
    free.measure("d", props.contact_depth(b))
    s2 = free.simulate(be)
    s2.step()
    assert np.isnan(s2["n"]).all() and np.isnan(s2["p"]).all() and s2["d"] == 0.0


@pytest.mark.parametrize("be", BACKENDS)
def test_disturbance_acts_only_inside_its_window(be):
    w = lobs.World(dt=0.002, substeps=5, gravity=(0, 0, 0))
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=2.0, pos=(0, 0, 1))
    w.add_disturbance(
        "gust", on=box, force=(0, 6.0, 8.0), start=0.2, duration=0.2
    )  # |F| = 10 N for 0.2 s
    w.measure("v", props.velocity(box))
    sim = w.simulate(be)
    sim.run(seconds=0.6)
    t, v = sim.history.time, sim.history["v"]
    assert np.abs(v[t < 0.2 - 1e-9]).max() < 1e-9  # nothing before the window
    assert np.allclose(
        v[-1], (0, 6.0 / 2 * 0.2, 8.0 / 2 * 0.2), rtol=0.03
    )  # impulse F*dt / m, then coasting
    assert np.allclose(v[t > 0.45], v[-1], atol=1e-9)  # nothing after
    with pytest.raises(lobs.errors.LobsError):
        w.add_disturbance("zero", on=box, force=(0, 0, 0))
    forever = lobs.World(dt=0.002, substeps=5, gravity=(0, 0, 0))
    b = forever.add_box("b", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    forever.add_disturbance("push", on=b, force=(2.0, 0, 0))
    forever.measure("vx", props.velocity(b).x)
    s = forever.simulate(be)
    s.run(seconds=0.5)
    assert s["vx"] == pytest.approx(1.0, rel=0.02)


def test_randomize_is_seeded_composable_and_validated():
    w = lobs.World(dt=0.002, gravity=(0, 0, 0))
    w.add_box("box", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    w.measure("pos", props.position(w.actor("box")))
    w.measure("heading", props.heading(w.actor("box")))
    spec = {"box": {"pos": (lobs.Uniform(-1, 1), lobs.Uniform(-1, 1), 1.0), "yaw": lobs.Uniform(-3, 3),
                    "linvel": lobs.Normal((0, 0, 0), 0.1)}}  # fmt: skip
    sim = w.simulate("mujoco")
    sim.reset(seed=3, randomize=spec)
    a = (sim["pos"].copy(), sim["heading"])
    sim.reset(seed=3, randomize=spec)
    assert np.array_equal(a[0], sim["pos"]) and a[1] == sim["heading"]  # reproducible
    sim.reset(seed=4, randomize=spec)
    assert not np.array_equal(a[0], sim["pos"])
    assert (
        -1 <= sim["pos"][0] <= 1
        and sim["pos"][2] == pytest.approx(1.0)
        and abs(sim["heading"]) <= 3.0 + 1e-9
    )
    # explicit initial_state beats the sampled value; distributions also work for joint states and choices
    sim.reset(seed=4, randomize=spec, initial_state={"box": lobs.Initial(pos=(5, 5, 5))})
    assert np.allclose(sim["pos"], (5, 5, 5))
    out = lobs.randomize.sample_initial(
        np.random.default_rng(0), {"box": {"q": {"j": lobs.Choice([0.1, 0.2])}}}
    )
    assert out["box"].q["j"] in (0.1, 0.2)
    with pytest.raises(ValueError, match="unknown fields"):
        lobs.randomize.sample_initial(np.random.default_rng(0), {"box": {"mass": 1}})


def test_gym_randomized_resets():
    w = lobs.World(dt=0.002, substeps=5, gravity=(0, 0, 0))
    b = w.add_box("box", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    b.set_initial(angvel=(0, 0, 0))
    w.measure("x", props.position(b).x)
    env = lobs.GymEnv(
        w, "mujoco", observe=["x"], randomize={"box": {"pos": (lobs.Uniform(-2, 2), 0.0, 1.0)}}
    )
    xs = {float(env.reset(seed=s)[0][0]) for s in range(5)}
    assert len(xs) == 5 and all(-2 <= x <= 2 for x in xs)
    assert float(env.reset(seed=2)[0][0]) == float(env.reset(seed=2)[0][0])


def test_merge_fixed_all_makes_one_static_actor(tmp_path):
    glb = make_gltf_room(tmp_path / "room.glb")
    prof = {"extends": "generic", "merge_fixed": "all"}
    world, report = lobs.import_world(glb, motion="auto", profile=prof, out=tmp_path / "m.lobs")
    statics = [a for a in world.spec.actors if a.name == "static_scene"]
    assert len(statics) == 1 and statics[0].root == "fixed" and len(statics[0].bodies) == 1
    ids = {g.source_id for g in statics[0].bodies[0].geoms}
    assert {"floor", "north_wall"} <= ids  # per-element identity survives the merge
    assert (
        next(a for a in world.spec.actors if a.source and a.source.source_id == "crate").root
        == "free"
    )
    assert lobs.check_world(world, "mujoco", seconds=1.0).passed
    unmerged, _ = lobs.import_world(glb, motion="auto", out=tmp_path / "u.lobs")
    assert len(world.spec.actors) < len(unmerged.spec.actors)


def test_cli_schema_prints_valid_json_schema(capsys):
    assert main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert (
        "forces" in schema["properties"] and "actors" in schema["properties"] and "$defs" in schema
    )
