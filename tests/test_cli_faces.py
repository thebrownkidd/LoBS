import numpy as np
import pytest
import yaml

import lobs
from lobs import math3d as m3
from lobs.cli import main
from lobs.faces import mesh_face_pose

from .fixtures.make_fixtures import make_gltf_room
from .test_mesh_shapes import cube


def test_mesh_face_pose_box_and_concave():
    v, f = cube(0.5)
    pos, quat = mesh_face_pose(v, f, "top")
    assert np.allclose(pos, (0, 0, 0.5), atol=1e-9) and np.allclose(
        m3.quat_to_mat(quat)[:, 2], (0, 0, 1)
    )
    pos, quat = mesh_face_pose(v, f, None, direction=(1, 1, 0), offset=0.1)
    # the ray hits the +x+y edge: one adjacent face wins, and the point is pushed out 0.1 along that face normal
    assert sorted(np.round(pos[:2], 6)) == [0.5, 0.6] and abs(pos[2]) < 1e-9
    pos, _ = mesh_face_pose(v, f, "left")
    assert np.allclose(pos, (0, 0.5, 0), atol=1e-9)
    import trimesh

    def pillar(x):
        return trimesh.creation.box(
            extents=(0.2, 1, 1), transform=trimesh.transformations.translation_matrix((x, 0, 0))
        )

    # two separated pillars: the centre column hits nothing, so the extreme triangle along +z is used
    two = trimesh.util.concatenate([pillar(-0.9), pillar(0.9)])
    pos, quat = mesh_face_pose(two.vertices, two.faces, "top")
    assert pos[2] == pytest.approx(0.5, abs=1e-6) and m3.quat_to_mat(quat)[2, 2] > 0.99
    assert abs(abs(pos[0]) - 0.9) < 0.11
    # a U with a base: looking down the centre line you see the base, not the arm tops
    base = trimesh.creation.box(
        extents=(2, 1, 0.2), transform=trimesh.transformations.translation_matrix((0, 0, -0.4))
    )
    u = trimesh.util.concatenate([pillar(-0.9), pillar(0.9), base])
    pos, _ = mesh_face_pose(u.vertices, u.faces, "top")
    assert pos[2] == pytest.approx(-0.3, abs=1e-6)
    with pytest.raises(lobs.errors.LobsError):
        mesh_face_pose(v, f, "sideways")


def test_cli_import_info_check_card_and_update(tmp_path, capsys):
    glb = make_gltf_room(tmp_path / "room.glb")
    pkg = tmp_path / "room.lobs"
    assert main(["import", str(glb), "-o", str(pkg), "--motion", "auto"]) == 0
    out = capsys.readouterr().out
    assert "imported" in out and "needs review" in out
    assert main(["info", str(pkg)]) == 0
    assert "actors" in capsys.readouterr().out
    assert main(["check", str(pkg), "--backend", "mujoco", "--seconds", "1.0"]) == 0
    assert "PASS" in capsys.readouterr().out
    card = yaml.safe_load((pkg / "card.yaml").read_text())
    assert card["validation"]["mujoco"]["passed"] is True
    assert main(["card", str(pkg)]) == 0
    # editing overrides then --update flips the decision
    ov = pkg / "import" / "overrides.yaml"
    ov.write_text(yaml.safe_dump({"elements": {"crate": {"motion": "fixed"}}}))
    assert main(["import", "--update", str(pkg)]) == 0
    capsys.readouterr()
    w = lobs.load_world(pkg)
    assert (
        next(a for a in w.spec.actors if a.source and a.source.source_id == "crate").root == "fixed"
    )
    # errors are reported cleanly with a non-zero exit code
    assert main(["import", str(glb), "-o", str(tmp_path / "x.lobs")]) == 2
    assert "motion" in capsys.readouterr().err
    assert (
        main(
            [
                "import",
                str(tmp_path / "missing.glb"),
                "-o",
                str(tmp_path / "y.lobs"),
                "--motion",
                "auto",
            ]
        )
        == 2
    )
