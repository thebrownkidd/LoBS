import numpy as np
import pytest
import yaml

import lobs
from lobs.errors import ImportFailure, LobsError

from .fixtures.make_fixtures import make_gltf_room, make_obj_mm

BACKENDS = lobs.backends.available()


@pytest.fixture
def room(tmp_path):
    return make_gltf_room(tmp_path / "room.glb")


def actor_by_source(world, source_id):
    return next(a for a in world.spec.actors if a.source and a.source.source_id == source_id)


def test_motion_mode_is_required(room, tmp_path):
    with pytest.raises(ImportFailure, match="motion"):
        lobs.import_world(room, out=tmp_path / "o.lobs")


def test_gltf_auto_mode_decisions_and_yup_conversion(room, tmp_path):
    world, report = lobs.import_world(room, motion="auto", out=tmp_path / "o.lobs")
    floor, crate = actor_by_source(world, "floor"), actor_by_source(world, "crate")
    assert floor.root == "fixed" and crate.root == "free"
    assert actor_by_source(world, "north_wall").root == "fixed"
    # glTF is Y-up: after import everything is Z-up, floor top at z=0, crate resting on it
    assert crate.pose.pos[2] == pytest.approx(0.2, abs=1e-6)
    assert crate.pose.pos[0] == pytest.approx(1.0, abs=1e-6)
    assert abs(floor.pose.pos[2] + 0.1) < 1e-6  # slab centre 0.1 below the surface
    # box shaped geometry is fitted exactly, the bracket is decomposed (non-convex)
    crate_geoms = [g for b in crate.bodies for g in b.geoms if g.role != "visual"]
    assert crate_geoms[0].shape.type == "box" and crate_geoms[0].shape.size == pytest.approx(
        (0.4, 0.4, 0.4)
    )
    bracket = actor_by_source(world, "bracket")
    assert any(g.shape.type == "convex_set" for b in bracket.bodies for g in b.geoms)
    # dynamic body mass = density * volume with an explicit inertial; fixed bodies carry none
    m = crate.bodies[0].inertial
    assert m.mode == "explicit" and 0 < m.mass == pytest.approx(
        0.4**3 * 1000.0, rel=1e-3
    )  # generic material 1000 kg/m^3
    assert floor.bodies[0].inertial.mass == 0.0
    # the log explains itself and flags guesses
    prov = {(d.target, d.field): d for d in report.decisions.items}
    assert prov[("crate", "motion")].provenance == "heuristic"
    assert report.needs_review  # crate guessed from its name, material defaulted ...


def test_gltf_dynamic_mode_keeps_ground_fixed(room, tmp_path):
    world, report = lobs.import_world(room, motion="dynamic", out=tmp_path / "o.lobs")
    assert actor_by_source(world, "floor").root == "fixed"  # ground support (low, wide slab)
    assert actor_by_source(world, "crate").root == "free"
    assert (
        actor_by_source(world, "north_wall").root == "free"
    )  # dynamic mode: everything else moves
    assert (
        next(
            d for d in report.decisions.items if d.target == "floor" and d.field == "motion"
        ).provenance
        == "heuristic"
    )
    assert not any(
        a.name == "ground" for a in world.spec.actors
    )  # a real ground was found: no plane added


def test_overrides_and_annotations_take_precedence(tmp_path):
    glb = make_gltf_room(
        tmp_path / "r.glb",
        extras={"crate": {"lobs:motion": "fixed", "lobs:mass": 3.0, "lobs:material": "steel"}},
    )
    world, report = lobs.import_world(glb, motion="dynamic", out=tmp_path / "o.lobs")
    crate = actor_by_source(world, "crate")
    assert crate.root == "fixed"  # source annotation beats --motion dynamic
    d = next(d for d in report.decisions.items if d.target == "crate" and d.field == "motion")
    assert d.provenance == "source"
    # an override beats the annotation, and survives a deterministic update
    (tmp_path / "o.lobs" / "import" / "overrides.yaml").write_text(
        yaml.safe_dump(
            {"elements": {"crate": {"motion": "dynamic", "mass": 5.0, "material": "wood"}}}
        )
    )
    world2, report2 = lobs.importers.api.update_world(tmp_path / "o.lobs")
    crate2 = actor_by_source(world2, "crate")
    assert crate2.root == "free" and crate2.bodies[0].inertial.mass == pytest.approx(5.0)
    assert (
        next(
            d for d in report2.decisions.items if d.target == "crate" and d.field == "material"
        ).value
        == "wood"
    )
    assert not report2.warnings or all("unknown source id" not in w for w in report2.warnings)


def test_rule_based_material_and_skip(room, tmp_path):
    prof = {"extends": "generic", "rules": [
        {"match": {"name": "north_*"}, "skip": True},
        {"match": {"name": "crate"}, "material": "steel", "motion": "dynamic"},
    ]}  # fmt: skip
    world, report = lobs.import_world(room, motion="auto", profile=prof, out=tmp_path / "o.lobs")
    assert not any(a.source and a.source.source_id == "north_wall" for a in world.spec.actors)
    crate = actor_by_source(world, "crate")
    assert crate.bodies[0].inertial.mass == pytest.approx(0.4**3 * 7850, rel=1e-3)
    assert (
        next(
            d for d in report.decisions.items if d.target == "crate" and d.field == "motion"
        ).provenance
        == "rule"
    )


def test_import_is_deterministic(room, tmp_path):
    lobs.import_world(room, motion="auto", out=tmp_path / "a.lobs")
    lobs.import_world(room, motion="auto", out=tmp_path / "b.lobs")
    for rel in ("world.yaml", "import/decisions.json", "import/profile.lock.yaml"):
        assert (tmp_path / "a.lobs" / rel).read_bytes() == (
            tmp_path / "b.lobs" / rel
        ).read_bytes(), rel
    assets_a = sorted(p.name for p in (tmp_path / "a.lobs" / "assets").iterdir())
    assets_b = sorted(p.name for p in (tmp_path / "b.lobs" / "assets").iterdir())
    assert assets_a == assets_b and assets_a  # content-addressed


def test_package_layout_and_reload(room, tmp_path):
    world, _ = lobs.import_world(room, motion="auto", out=tmp_path / "o.lobs", copy_source=True)
    root = tmp_path / "o.lobs"
    for rel in ("world.yaml", "card.yaml", "assets", "import/decisions.json", "import/report.md",
                "import/overrides.yaml", "import/profile.lock.yaml", "source/room.glb"):  # fmt: skip
        assert (root / rel).exists(), rel
    again = lobs.load_world(root)
    assert again.spec == world.spec and again.base_dir == root
    assert again.spec.imports[0].motion_mode == "auto" and again.spec.imports[0].source_sha256
    card = yaml.safe_load((root / "card.yaml").read_text())
    assert (
        card["counts"]["actors"] == len(world.spec.actors)
        and "rigid_bodies" in card["capabilities_required"]
    )


@pytest.mark.parametrize("be", BACKENDS)
def test_imported_room_is_stable_and_crate_rests(room, tmp_path, be):
    world, _ = lobs.import_world(room, motion="auto", out=tmp_path / "o.lobs")
    crate = actor_by_source(world, "crate")
    world.measure("crate_h", lobs.props.height(world.actor(crate.name)))
    report = world.validate(be)
    if not report.ok:  # e.g. an engine that cannot do something in the file: must say so clearly
        pytest.skip(report.format())
    res = lobs.check_world(world, be, seconds=1.5)
    assert res.passed, res.summary()
    sim = world.simulate(be)
    sim.run(seconds=1.5)
    assert sim["crate_h"] == pytest.approx(0.2, abs=6e-3)


def test_obj_units_and_missing_units_flag(tmp_path):
    obj = make_obj_mm(tmp_path / "scene.obj")
    world, report = lobs.import_world(obj, motion="dynamic", units="mm", out=tmp_path / "mm.lobs")
    assert next(d for d in report.decisions.items if d.field == "units").provenance == "override"
    sizes = [
        g.shape.size
        for a in world.spec.actors
        for b in a.bodies
        for g in b.geoms
        if g.shape.type == "box"
    ]
    assert sorted(round(max(s), 6) for s in sizes) == [0.4, 4.0]  # mm -> m: cube 0.4 m, slab 4 m
    # without units the format gives us nothing: assumed metres, loudly, with a millimetre hint
    _, report2 = lobs.import_world(obj, motion="dynamic", out=tmp_path / "guess.lobs")
    d = next(d for d in report2.decisions.items if d.field == "units")
    assert d.provenance == "default" and d.review and "millimetres" in d.rationale


def test_huge_coordinates_are_recentred_with_georeference(tmp_path):
    import trimesh

    far = trimesh.creation.box(extents=(4, 4, 1))
    far.apply_translation((500000.0, 4000000.0, 100.0))
    obj = tmp_path / "far.obj"
    far.export(obj)
    world, report = lobs.import_world(obj, motion="dynamic", units="m", out=tmp_path / "far.lobs")
    pos = world.spec.actors[0].pose.pos
    assert max(abs(x) for x in pos) < 10
    assert world.spec.georeference is not None
    off = world.spec.georeference.offset_m
    assert off[0] == pytest.approx(500000.0) and off[1] == pytest.approx(4000000.0)


def test_include_merges_worlds_and_rewrites_measures(room, tmp_path):
    guest, _ = lobs.import_world(room, motion="auto", out=tmp_path / "g.lobs")
    guest.measure("h", lobs.props.height(guest.actor("crate")))
    guest.save(tmp_path / "g.lobs")
    host = lobs.World("host")
    host.add_box("probe", (0.1, 0.1, 0.1), mass=1.0, pos=(0, 0, 3))
    host.include(tmp_path / "g.lobs", prefix="room", pose=lobs.Pose(pos=(10, 0, 0)))
    assert host.actor("room.crate")
    assert host.registry().bodies[-1].path.startswith("room.")
    assert "room.h" in host.spec.measures
    # the guest's assets were copied into the host and the merged world simulates
    assert host.base_dir is not None and any((host.base_dir / "assets").iterdir())
    host.measure(
        "probe_to_crate", lobs.rel.between(host.actor("probe"), host.actor("room.crate")).distance()
    )
    sim = host.simulate("mujoco")
    sim.run(seconds=0.2)
    assert sim["room.h"] == pytest.approx(0.2, abs=1e-3) and sim["probe_to_crate"] > 5
    with pytest.raises(LobsError):
        host.include(tmp_path / "g.lobs", prefix="room")  # name clash


def test_mesh_anchor_on_face_raycast(room, tmp_path):
    world, _ = lobs.import_world(room, motion="auto", out=tmp_path / "o.lobs")
    crate = world.actor(actor_by_source(world, "crate").name)
    # the crate's collision geom is a box here; use the (convex-hulled) bracket for the mesh path instead
    bracket = world.actor(actor_by_source(world, "bracket").name)
    a = bracket.anchor_on_face("top", name="tip")
    reg = world.registry()
    anchor = next(x for x in reg.anchors if x.path.endswith("/tip"))
    n = lobs.math3d.quat_to_mat(anchor.quat)[:, 2]
    assert n[2] > 0.9  # outward normal pointing up
    assert a.path.endswith("/tip") and crate.path
    assert np.isfinite(anchor.pos).all()


def test_check_names_the_overlapping_pair_and_the_exploding_body(tmp_path):
    """A crate imported inside a building must be reported as such, not just as a mystery explosion."""
    import trimesh

    scene = trimesh.Scene()
    slab = trimesh.creation.box(extents=(20, 0.5, 20))
    slab.apply_translation((0, -0.25, 0))
    wall = trimesh.creation.box(extents=(4, 3, 4))
    wall.apply_translation((0, 1.5, 0))
    crate = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
    crate.apply_translation((0.5, 1.5, 0.5))  # buried inside the wall block
    for name, g in (("floor", slab), ("building", wall), ("crate", crate)):
        scene.add_geometry(g, node_name=name, geom_name=name)
    glb = tmp_path / "overlap.glb"
    glb.write_bytes(scene.export(file_type="glb"))
    world, _ = lobs.import_world(glb, motion="auto", out=tmp_path / "o.lobs")
    res = lobs.check_world(world, "mujoco", seconds=1.0)
    assert not res.passed
    text = res.summary()
    assert "interpenetrating" in text and "crate" in text and "building" in text
    assert res.initial_penetrations and res.initial_penetrations[0][1] > 0.5
    # the same world without the overlap passes, and a looser limit tolerates small overlaps
    clean = lobs.World("clean")
    clean.add_plane("ground")
    clean.add_box("a", (0.4, 0.4, 0.4), mass=1.0, pos=(0, 0, 0.19))  # 1 cm into the plane
    assert lobs.check_world(clean, "mujoco", seconds=0.5).passed
    assert not lobs.check_world(clean, "mujoco", seconds=0.5, penetration_limit=0.005).passed
