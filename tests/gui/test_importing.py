import pytest
import trimesh

from lobs.gui.core import importing as imp


def make_glb(tmp_path):
    scene = trimesh.Scene()
    for name, extents, centre in (
        ("floor", (10, 0.2, 10), (0, -0.1, 0)),
        ("wooden_crate", (0.4, 0.4, 0.4), (1.0, 0.2, 0.5)),
        ("north_wall", (10, 3, 0.2), (0, 1.5, -5)),
    ):
        m = trimesh.creation.box(extents=extents)
        m.apply_translation(centre)
        scene.add_geometry(m, node_name=name, geom_name=name)
    p = tmp_path / "room.glb"
    p.write_bytes(scene.export(file_type="glb"))
    return p


def test_motion_mode_is_required_then_import_decisions_and_overrides(tmp_path):
    src = make_glb(tmp_path)
    req = imp.ImportRequest(str(src), out=str(tmp_path / "room.lobs"))
    assert not req.ready()
    with pytest.raises(ValueError):
        imp.run_import(req)
    req.motion = "auto"
    world, report = imp.run_import(req)
    assert report.counts["actors"] >= 3
    rows = imp.decision_rows(report)
    assert rows and all({"target", "field", "provenance", "confidence"} <= set(r) for r in rows)
    assert len(imp.decision_rows(report, only_review=True)) <= len(rows)
    crate = next(r for r in rows if r["field"] == "motion" and "crate" in r["target"].lower())
    # make the crate heavy via an override; the re-import must honour it
    put = imp.set_override
    put(req.overrides, crate["target"], "mass", imp.parse_value("mass", "123.5"))
    world2, report2 = imp.run_import(req)
    masses = [d.value for d in report2.decisions.items if d.target == crate["target"] and d.field == "mass"]
    assert masses and masses[-1] == pytest.approx(123.5)
    with pytest.raises(ValueError):
        put(req.overrides, crate["target"], "pose", 1)
    imp.clear_override(req.overrides, crate["target"])
    assert not req.overrides
    assert "glb" in imp.file_filter()
