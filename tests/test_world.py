import numpy as np
import pytest

import lobs
from lobs import math3d as m3
from lobs import props, rel
from lobs.errors import LobsError
from lobs.kinematics import fk_actor
from lobs.registry import build_registry, resolve_initial


def pendulum_world():
    w = lobs.World("pend", dt=0.001)
    pend = w.add_actor("pend", root="fixed", pos=(0, 0, 2))
    base = pend.add_body("base", geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0)
    base.add_child(
        "rod",
        pos=(0, 0, -0.5),
        joint=lobs.Hinge("hinge", axis=(0, 1, 0), pos=(0, 0, 0.5)),
        geoms=[lobs.Capsule(radius=0.02, length=1.0)],
        mass=0.5,
    )
    pend.add_actuator("motor", joint="hinge", kind="effort", ctrl_range=(-5, 5))
    return w, pend


def test_static_dynamic_split_and_masses():
    w = lobs.World()
    w.add_plane("ground")
    w.add_box("fixedbox", (1, 1, 1), fixed=True)
    w.add_box("box", (0.2, 0.2, 0.2), mass=2.0, pos=(0, 0, 1))
    reg = w.registry()
    assert [b.path for b in reg.bodies] == ["ground/body", "fixedbox/body", "box/body"]
    assert [b.static for b in reg.bodies] == [True, True, False]
    assert reg.dynamic_order == [2]
    assert reg.bodies[2].dyn == 0 and reg.bodies[0].dyn == -1
    assert reg.bodies[2].mass == pytest.approx(2.0)
    assert reg.bodies[1].mass == pytest.approx(1000.0)  # default density
    assert np.allclose(reg.bodies[1].world_pos, 0)


def test_joint_under_fixed_root_makes_child_dynamic_but_weld_stays_static():
    w, pend = pendulum_world()
    base = pend.body("base")
    base.add_child("plate", pos=(1, 0, 0), geoms=[lobs.Box(size=(0.1, 0.1, 0.1))], mass=1.0)
    reg = w.registry()
    by = {b.name: b for b in reg.bodies}
    assert by["base"].static and by["plate"].static and not by["rod"].static
    assert np.allclose(by["plate"].world_pos, (1, 0, 2))


def test_duplicate_and_structural_errors():
    w = lobs.World()
    w.add_box("a", (1, 1, 1))
    with pytest.raises(LobsError):
        w.add_box("a", (1, 1, 1))
    with pytest.raises(LobsError):
        w.add_actor("world")
    w2 = lobs.World()
    act = w2.add_actor("dyn", root="free")
    act.add_body("b", geoms=[lobs.Plane()])
    assert not w2.validate().ok  # plane on a dynamic body


def test_fk_pendulum_matches_analytic():
    w, _ = pendulum_world()
    reg = w.registry()
    q, qd = 0.7, 1.3
    res = fk_actor(
        reg, 0, w.spec.actors[0].pose, np.array([q]), np.array([qd]), np.zeros(3), np.zeros(3)
    )
    p, quat, v, om = res[1]
    assert np.allclose(p, (-0.5 * np.sin(q), 0, 2 - 0.5 * np.cos(q)))
    assert np.allclose(v, (-0.5 * qd * np.cos(q), 0, 0.5 * qd * np.sin(q)))
    assert np.allclose(om, (0, qd, 0))
    assert np.allclose(m3.quat_to_mat(quat), m3.quat_to_mat(m3.quat_from_axis_angle([0, 1, 0], q)))


def test_fk_slide_and_moving_parent():
    w = lobs.World()
    a = w.add_actor("rail", root="free", pos=(1, 0, 0))
    base = a.add_body("base", geoms=[lobs.Box(size=(1, 1, 1))], mass=1)
    base.add_child("car", pos=(0, 0, 0), joint=lobs.Slide("s", axis=(1, 0, 0)),
                   geoms=[lobs.Box(size=(1, 1, 1))], mass=1)  # fmt: skip
    reg = w.registry()
    w_ = np.array([0, 0, 2.0])
    res = fk_actor(reg, 0, w.spec.actors[0].pose, np.array([0.5]), np.array([0.25]),
                   np.array([1.0, 0, 0]), w_)  # fmt: skip
    p, _, v, om = res[1]
    # parent frame rotated 0 (identity quat); slide along x by 0.5 from origin at (1,0,0)
    assert np.allclose(p, (1.5, 0, 0))
    # v = parent v + w x (child offset) + qd along axis
    assert np.allclose(v, np.array([1.0, 0, 0]) + np.cross(w_, [0.5, 0, 0]) + [0.25, 0, 0])
    assert np.allclose(om, w_)


def test_face_anchors():
    w = lobs.World()
    b = w.add_box("b", (2, 4, 6), mass=1)
    top = b.anchor_on_face("top")
    b.anchor_on_face("+x", name="px", uv=(1.0, 0.0))
    reg = w.registry()
    a = {x.path: x for x in reg.anchors}
    assert np.allclose(a["b/top"].pos, (0, 0, 3))
    assert np.allclose(m3.quat_to_mat(a["b/top"].quat)[:, 2], (0, 0, 1))
    n = m3.quat_to_mat(a["b/px"].quat)[:, 2]
    assert np.allclose(n, (1, 0, 0))
    # +x face of a 2x4x6 box; u runs along the face's first tangent (+y here), scaled by its half-extent
    assert np.allclose(a["b/px"].pos, (1, 2, 0))
    assert top.path == "b/top"
    with pytest.raises(LobsError):
        b.anchor_on_face("top")  # duplicate name

    c = w.add_cylinder("c", radius=0.5, length=2, mass=1)
    c.anchor_on_face("top")
    c.anchor_on_face("side", name="s", angle=np.pi / 2)
    s = w.add_sphere("s", radius=1.0, mass=1)
    s.anchor_on_face(name="north", direction=(0, 0, 2))
    reg = w.registry()
    a = {x.path: x for x in reg.anchors}
    assert np.allclose(a["c/top"].pos, (0, 0, 1))
    assert np.allclose(a["c/s"].pos, (0, 0.5, 0), atol=1e-12)
    assert np.allclose(a["s/north"].pos, (0, 0, 1))


@pytest.mark.parametrize("name", ["w.yaml", "w.json", "w.lobs"])
def test_save_load_roundtrip(tmp_path, name):
    w, pend = pendulum_world()
    w.add_plane("ground")
    box = w.add_box("box", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    top = box.anchor_on_face("top")
    goal = w.anchor("goal", pos=(1, 0, 0.1), rule=lobs.Path.circle(0.5, 4.0))
    w.measure("h", props.height(box))
    g = w.measure("gap", rel.between(top, goal).distance())
    w.measure("closing", g.rate())
    w.save(tmp_path / name)
    w2 = lobs.load_world(tmp_path / name)
    assert w2.spec == w.spec
    assert w2.validate().ok


def test_resolve_initial_overrides():
    w, _ = pendulum_world()
    w.add_box("box", (1, 1, 1), mass=1, linvel=(1, 2, 3))
    reg = w.registry()
    ri = resolve_initial(
        reg, {"pend": lobs.Initial(q={"hinge": 0.3}), "box": lobs.Initial(pos=(5, 0, 0))}
    )
    assert ri.q[0] == pytest.approx(0.3)
    assert np.allclose(ri.actor_pose[1][0], (5, 0, 0))
    assert np.allclose(ri.linvel[1], (1, 2, 3))
    with pytest.raises(LobsError):
        resolve_initial(reg, {"nope": lobs.Initial()})


def test_find_by_tag_and_resolve_paths():
    w = lobs.World()
    w.add_box("a", (1, 1, 1), tags=["storey:1"])
    w.add_box("b", (1, 1, 1))
    assert [h.path for h in w.find(tag="storey:1")] == ["a"]
    reg = build_registry(w.spec)
    assert reg.resolve("a:origin").point == "origin"
    assert reg.resolve(lobs.WORLD).kind == "world"
    with pytest.raises(LobsError):
        reg.resolve("a/nope")
    with pytest.raises(LobsError):
        reg.resolve("a:bogus")
