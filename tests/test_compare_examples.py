import os
import runpy
from pathlib import Path

import numpy as np
import pytest

import lobs
from lobs import props

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))
BOTH = set(lobs.backends.available()) >= {"mujoco", "chrono"}


def pendulum_world():
    w = lobs.World("pend", dt=0.002)
    p = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = p.add_body("base", geoms=[lobs.Box(size=(0.05, 0.05, 0.05))], mass=1.0)
    base.add_child("rod", pos=(0, 0, -0.5), joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5)),
                   geoms=[lobs.Capsule(radius=0.02, length=1.0)], mass=0.5, collision_group=7)  # fmt: skip
    w.spec.contact.disabled_group_pairs.append((7, 7))
    p.set_initial(q={"hinge": 0.6})
    w.measure("theta", props.joint_position(p.joint("hinge")))
    w.measure("energy", props.mechanical_energy(p.body("rod")))
    return w


@pytest.mark.skipif(not BOTH, reason="needs mujoco and chrono")
def test_compare_pendulum_engines_agree_closely():
    comp = lobs.compare(pendulum_world(), ["mujoco", "chrono"], seconds=3.0)
    assert not comp.errors
    m = comp.metrics["theta"]["chrono"]
    assert m.rmse < 0.05 and m.max_abs < 0.1  # same physics, different integrators
    assert comp.runtime_per_tick["mujoco"] > 0 and "friction" in comp.capability_differences
    assert comp.approximations["chrono"]  # approximations are reported, not hidden
    text = comp.table()
    assert "theta" in text and "reference: mujoco" in text


def test_compare_reports_unrunnable_backend():
    w = lobs.World()
    w.add_actor("m", root="fixed").add_body("b", geoms=[lobs.schema.TriangleMesh(file="x.obj")])
    w.add_box("b", (1, 1, 1), mass=1.0)
    comp = lobs.compare(w, ["mujoco", "mujoco"], ticks=1)
    assert set(comp.errors) == {"mujoco"}
    with pytest.raises(lobs.errors.LobsError):
        lobs.compare(w, ["mujoco"])


def test_compare_same_backend_twice_is_identical():
    comp = lobs.compare(pendulum_world(), ["mujoco", "mujoco"], seconds=0.5)
    assert comp.metrics["theta"]["mujoco"].max_abs == 0.0
    assert np.isfinite(comp.metrics["energy"]["mujoco"].rmse)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_run(path, tmp_path, capsys):
    if path.name.startswith("04") and not BOTH:
        pytest.skip("needs both engines")
    old = Path.cwd()
    os.chdir(tmp_path)
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        os.chdir(old)
    assert capsys.readouterr().out.strip()
