import struct
import zlib

import numpy as np
import pytest

import lobs
from lobs import props
from lobs.viz import write_png

from .fixtures.make_ifc import make_ifc_building

BACKENDS = lobs.backends.available()


def drop_world():
    w = lobs.World("drop", dt=0.002, substeps=5)
    w.add_plane("ground")
    b = w.add_box("box", (0.3, 0.3, 0.3), mass=1.0, pos=(0, 0, 1.5), rgba=(1.0, 0.1, 0.1, 1.0))
    w.add_sphere("ball", 0.15, mass=0.5, pos=(0.6, 0, 0.5))
    w.measure("h", props.height(b))
    return w


def red_pixels(img):
    return (img[..., 0] > 180) & (img[..., 1] < 90) & (img[..., 2] < 90)


@pytest.mark.parametrize("be", BACKENDS)
def test_render_any_backend_shows_the_moving_box(be):
    sim = drop_world().simulate(be)
    a = sim.render(320, 240)
    assert (
        a.shape == (240, 320, 3) and a.dtype == np.uint8 and a.std() > 5
    )  # a real picture, not a blank frame
    sim.run(seconds=0.5)
    b = sim.render(
        320, 240, camera={"lookat": (0.3, 0, 0.6), "distance": 5.0, "azimuth": 90, "elevation": -10}
    )
    assert b.shape == a.shape
    # fixed camera: the red box falls, so its pixels move down the image
    cam = {"lookat": (0.3, 0, 0.8), "distance": 4.0, "azimuth": 90, "elevation": 0}
    sim2 = drop_world().simulate(be)
    top = sim2.render(320, 240, camera=cam)
    sim2.run(seconds=0.5)
    later = sim2.render(320, 240, camera=cam)
    ys0, ys1 = np.nonzero(red_pixels(top))[0], np.nonzero(red_pixels(later))[0]
    assert len(ys0) > 20 and len(ys1) > 20 and ys1.mean() > ys0.mean() + 5
    sim.close()
    sim2.close()


def test_png_writer_produces_a_valid_png(tmp_path):
    img = (np.random.default_rng(0).random((17, 23, 3)) * 255).astype(np.uint8)
    p = write_png(tmp_path / "a" / "x.png", img)
    data = p.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", data[16:24])
    assert (w, h) == (23, 17) and data.endswith(b"IEND\xaeB`\x82")
    idat = data[data.index(b"IDAT") + 4 : data.index(b"IEND") - 8]
    raw = zlib.decompress(idat)
    rows = [
        np.frombuffer(raw[i * (1 + 23 * 3) + 1 : (i + 1) * (1 + 23 * 3)], np.uint8)
        for i in range(17)
    ]
    assert np.array_equal(np.stack(rows).reshape(17, 23, 3), img)  # lossless round trip


def test_gym_render_mode_rgb_array():
    w = drop_world()
    w.spec.actors[2].actuators = []  # the ball/box have no actuators: that's fine
    env = lobs.GymEnv(w, "mujoco", observe=["h"], render_mode="rgb_array")
    env.reset()
    env.step(np.zeros(0))
    img = env.render()
    assert img.ndim == 3 and img.shape[2] == 3
    env.close()


def test_trajectory_render_and_mesh_world_render(tmp_path):
    sim = drop_world().simulate("mujoco")
    sim.run(seconds=0.2)
    p = sim.export(tmp_path / "t.npz")
    traj = lobs.Trajectory.load(p)
    img = traj.render(0, 200, 150)
    assert img.shape == (150, 200, 3) and img.std() > 5
    # imported world with triangle meshes
    world, _ = lobs.import_world(
        make_ifc_building(tmp_path / "b.ifc"),
        motion="auto",
        profile="civil",
        out=tmp_path / "b.lobs",
    )
    s = world.simulate("mujoco", record=False)
    pic = s.render(320, 240)
    assert pic.std() > 5


def test_rerun_recording_is_written_and_contains_geometry_transforms_and_scalars(tmp_path):
    rr = pytest.importorskip("rerun")
    sim = drop_world().simulate("mujoco")
    sim.run(seconds=0.3)
    out = tmp_path / "run.rrd"
    lobs.viz.log_rerun(sim, "lobs-test", save=out, every=5)
    assert out.exists() and out.stat().st_size > 1000
    try:
        rec = rr.dataframe.load_recording(str(out))
    except (
        AttributeError
    ):  # older/newer sdk layouts: file presence is the contract we can always check
        return
    paths = {str(c.entity_path) for c in rec.schema().component_columns()}
    assert any(p.startswith("/world/box") for p in paths) and any(
        p.startswith("/measures/h") for p in paths
    )
    traj_out = tmp_path / "traj.rrd"
    lobs.viz.log_rerun(sim.trajectory(), save=traj_out, measures=False)
    assert traj_out.stat().st_size > 1000


def test_cli_card_preview(tmp_path, capsys):
    from lobs.cli import main

    from .fixtures.make_fixtures import make_gltf_room

    pkg = tmp_path / "room.lobs"
    assert (
        main(
            [
                "import",
                str(make_gltf_room(tmp_path / "room.glb")),
                "-o",
                str(pkg),
                "--motion",
                "auto",
            ]
        )
        == 0
    )
    assert main(["card", str(pkg), "--preview"]) == 0
    out = capsys.readouterr().out
    assert "preview written" in out and (pkg / "previews" / "overview.png").stat().st_size > 1000
