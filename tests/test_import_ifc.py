import pytest

import lobs

from .fixtures.make_ifc import make_ifc_building

BACKENDS = lobs.backends.available()


def by_class(world, cls):
    return [a for a in world.spec.actors if a.source and a.source.semantic_class == cls]


@pytest.fixture(scope="module")
def ifc(tmp_path_factory):
    return make_ifc_building(tmp_path_factory.mktemp("ifc") / "building.ifc")


@pytest.fixture(scope="module")
def auto_world(ifc, tmp_path_factory):
    return lobs.import_world(
        ifc, motion="auto", profile="civil", out=tmp_path_factory.mktemp("o") / "b.lobs"
    )


def test_ifc_units_and_geometry_in_metres(auto_world):
    world, report = auto_world
    unit = next(d for d in report.decisions.items if d.field == "units")
    assert unit.provenance == "source" and unit.value.startswith("0.001")
    chair = by_class(world, "IfcFurniture")[0]
    # 450 x 900(high) x 450 mm chair at (3000, 3000, 200) mm: centre (3.225, 3.225, 0.65) m
    assert chair.pose.pos == pytest.approx((3.225, 3.225, 0.65), abs=1e-6)
    sizes = sorted(
        round(x, 4) for x in next(g for g in chair.bodies[0].geoms if g.role != "visual").shape.size
    )
    assert sizes == [0.45, 0.45, 0.9]


def test_civil_profile_auto_motion_and_semantics(auto_world):
    world, report = auto_world
    assert all(a.root == "fixed" for a in by_class(world, "IfcWall"))
    assert (
        by_class(world, "IfcSlab")[0].root == "fixed"
        and by_class(world, "IfcColumn")[0].root == "fixed"
    )
    assert (
        by_class(world, "IfcDoor")[0].root == "fixed"
    )  # civil profile: doors are structural until hinges are modelled
    chair = by_class(world, "IfcFurniture")[0]
    assert chair.root == "free"
    assert (
        "storey:Level 0" in chair.tags
        and "building:Main building" in chair.tags
        and "class:IfcFurniture" in chair.tags
    )
    d = next(
        d
        for d in report.decisions.items
        if d.target == chair.source.source_id and d.field == "material"
    )
    assert d.value == "wood" and d.provenance in ("rule", "material-library")
    assert chair.bodies[0].inertial.mass == pytest.approx(0.45 * 0.45 * 0.9 * 600.0, rel=1e-3)
    col = by_class(world, "IfcColumn")[0]
    # the civil profile's class rule sets columns to reinforced concrete (rule beats the IFC material name)
    assert (
        next(
            d
            for d in report.decisions.items
            if d.target == col.source.source_id and d.field == "material"
        ).value
        == "reinforced_concrete"
    )
    assert not any(a.name == "ground" for a in world.spec.actors)  # the slab is the ground support


def test_ifc_dynamic_mode_moves_everything_but_ground(ifc, tmp_path):
    world, _ = lobs.import_world(ifc, motion="dynamic", profile="civil", out=tmp_path / "d.lobs")
    assert by_class(world, "IfcSlab")[0].root == "fixed"
    assert all(a.root == "free" for a in by_class(world, "IfcWall"))
    assert by_class(world, "IfcFurniture")[0].root == "free"


def test_pset_lobs_annotation_is_source_provenance(tmp_path):
    ifc = make_ifc_building(
        tmp_path / "a.ifc", pset_lobs={"collision": "convex_hull", "motion": "dynamic"}
    )
    world, report = lobs.import_world(ifc, motion="auto", profile="civil", out=tmp_path / "a.lobs")
    wall = next(a for a in by_class(world, "IfcWall") if a.name == "Wall_South")
    assert wall.root == "free"  # authored in the BIM tool: beats the structural-class heuristic
    d = [d for d in report.decisions.items if d.target == wall.source.source_id]
    assert next(x for x in d if x.field == "motion").provenance == "source"
    assert next(x for x in d if x.field == "collision").provenance == "source"


def test_ifc_far_from_origin_is_recentred(tmp_path):
    ifc = make_ifc_building(
        tmp_path / "far.ifc", offset_mm=(5.0e8, 4.0e9, 0.0)
    )  # 500 km / 4000 km in mm
    world, report = lobs.import_world(
        ifc, motion="auto", profile="civil", out=tmp_path / "far.lobs"
    )
    assert max(abs(x) for a in world.spec.actors for x in a.pose.pos) < 20
    assert world.spec.georeference is not None
    assert world.spec.georeference.offset_m[0] == pytest.approx(5.0e5, rel=1e-3)
    assert any(d.field == "origin" for d in report.decisions.items)


@pytest.mark.parametrize("be", BACKENDS)
def test_ifc_building_simulates_and_chair_rests(auto_world, be):
    world, _ = auto_world
    chair = by_class(world, "IfcFurniture")[0]
    if "chair_h" not in world.spec.measures:
        world.measure("chair_h", lobs.props.height(world.actor(chair.name)))
    report = world.validate(be)
    if not report.ok:
        pytest.skip(report.format())
    res = lobs.check_world(world, be, seconds=1.5)
    assert res.passed, res.summary()
    sim = world.simulate(be)
    sim.run(seconds=1.5)
    assert sim["chair_h"] == pytest.approx(
        0.65, abs=1e-2
    )  # rests on the slab top (z=0.2) + half height


def test_profile_force_laws_select_by_class(ifc, tmp_path):
    profile = {
        "extends": "civil",
        "forces": [
            {
                "name": "wind",
                "select": {"class": "IfcFurniture"},
                "direction": [1, 0, 0],
                "acceleration": 0.5,
            }
        ],
    }
    world, report = lobs.import_world(ifc, motion="auto", profile=profile, out=tmp_path / "w.lobs")
    assert [f.name for f in world.spec.forces] == [
        "wind.Office_chair"
    ]  # only the chair: the only furniture
    chair = by_class(world, "IfcFurniture")[0]
    mass = chair.bodies[0].inertial.mass
    f = world.spec.forces[0]
    assert (
        f.magnitude.value == pytest.approx(0.5 * mass)
        and f.direction == (1.0, 0.0, 0.0)
        and not f.reaction
    )
    assert any(d.field == "force" and d.provenance == "rule" for d in report.decisions.items)
    world.measure("vx", lobs.props.velocity(world.actor(chair.name)).x)
    sim = world.simulate("mujoco")
    sim.run(seconds=0.2)
    assert sim["vx"] > 0.0  # pushed along +x (friction with the slab limits it, but it moves)
    with pytest.raises(lobs.errors.ImportFailure, match="force entries"):
        lobs.import_world(
            ifc,
            motion="auto",
            profile={"extends": "civil", "forces": [{"name": "bad"}]},
            out=tmp_path / "e.lobs",
        )
