import numpy as np
import pytest

import lobs
from lobs import props
from lobs.assets import load_mesh, read_obj, store_mesh
from lobs.schema import ConvexMesh, ConvexSet, GeomSpec, InertialSpec, TriangleMesh

BACKENDS = lobs.backends.available()


def cube(h=0.1):
    v = np.array([[x, y, z] for x in (-h, h) for y in (-h, h) for z in (-h, h)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])  # fmt: skip
    return v, f


def slab(half=5.0, thick=0.5):
    v, f = cube(1.0)
    v = v * np.array([half, half, thick])
    return v, f


def inertial_cube(mass=1.0, h=0.1):
    i = np.eye(3) * mass * (2 * h) ** 2 / 6
    return InertialSpec(mode="explicit", mass=mass, com=(0, 0, 0), inertia=tuple(map(tuple, i)))


def mesh_world(tmp_path, shape_kind):
    pkg = tmp_path / "w.lobs"
    w = lobs.World("mesh", dt=0.002, substeps=2)
    w.add_plane("ground")
    v, f = cube()
    rel = store_mesh(pkg / "assets", v, f)
    a = w.add_actor("m", root="free", pos=(0, 0, 0.6))
    shape = ConvexMesh(file=rel) if shape_kind == "mesh" else ConvexSet(files=[rel])
    a.add_body("b", geoms=[GeomSpec(shape=shape)], inertial=inertial_cube())
    w.measure("h", props.height(a))
    w.save(pkg)
    return lobs.load_world(pkg)


def test_obj_roundtrip_and_content_addressing(tmp_path):
    v, f = cube()
    r1 = store_mesh(tmp_path, v, f)
    r2 = store_mesh(tmp_path, v.copy(), f.copy())
    assert r1 == r2 and r1.startswith("assets/")
    v2, f2 = read_obj(tmp_path / r1.split("/")[1])
    assert np.allclose(v, v2) and np.array_equal(f, f2)
    with pytest.raises(lobs.errors.LobsError):
        load_mesh(None, "assets/x.obj")


@pytest.mark.parametrize("be", BACKENDS)
@pytest.mark.parametrize("kind", ["mesh", "set"])
def test_convex_mesh_rests_on_ground(tmp_path, be, kind):
    w = mesh_world(tmp_path, kind)
    assert w.base_dir is not None
    sim = w.simulate(be)
    sim.run(seconds=1.5)
    assert sim["h"] == pytest.approx(0.1, abs=4e-3)  # cube half-extent
    if be == "mujoco":  # honest reporting: hull collision is flagged as approximate
        assert any("convex hull" in i.message for i in w.validate(be).warnings)


@pytest.mark.parametrize("be", BACKENDS)
def test_static_triangle_mesh_floor(tmp_path, be):
    pkg = tmp_path / "t.lobs"
    w = lobs.World("tm", dt=0.002, substeps=2)
    v, f = slab()
    rel = store_mesh(pkg / "assets", v, f)
    floor = w.add_actor("floor", root="fixed", pos=(0, 0, -0.5))
    floor.add_body("b", geoms=[GeomSpec(shape=TriangleMesh(file=rel))],
                   inertial=InertialSpec(mode="explicit", mass=0.0, inertia=((0, 0, 0),) * 3))  # fmt: skip
    cv, cf = cube()
    crel = store_mesh(pkg / "assets", cv, cf)
    a = w.add_actor("m", root="free", pos=(0, 0, 0.6))
    a.add_body("b", geoms=[GeomSpec(shape=ConvexMesh(file=crel))], inertial=inertial_cube())
    w.measure("h", props.height(a))
    w.save(pkg)
    w = lobs.load_world(pkg)
    report = w.validate(be)
    if be == "mujoco":
        assert not report.ok and "shape_triangle_mesh_static" in report.format()
        return
    sim = w.simulate(be)
    sim.run(seconds=1.5)
    assert sim["h"] == pytest.approx(0.1, abs=4e-3)  # floor top at z=0


def test_visual_meshes_are_skipped_by_backends(tmp_path):
    pkg = tmp_path / "v.lobs"
    w = lobs.World("vis", dt=0.002)
    v, f = cube()
    rel = store_mesh(pkg / "assets", v, f)
    a = w.add_actor("m", root="free", pos=(0, 0, 1))
    a.add_body("b", geoms=[GeomSpec(shape=lobs.Box(size=(0.2, 0.2, 0.2))),
                           GeomSpec(shape=TriangleMesh(file=rel), role="visual")], mass=1.0)  # fmt: skip
    w.save(pkg)
    for be in BACKENDS:
        assert lobs.load_world(pkg).validate(be).ok
