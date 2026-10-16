"""A small BIM model generated with IfcOpenShell (no binary fixtures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def make_ifc_building(
    path: Path, offset_mm=(0.0, 0.0, 0.0), unit="MILLIMETERS", pset_lobs=None
) -> Path:
    """A 10 x 10 m, one-storey building in millimetres: slab, four walls, a column, a door, a wooden chair."""
    import ifcopenshell
    import ifcopenshell.api

    run = ifcopenshell.api.run
    f = run("project.create_file")
    project = run("root.create_entity", f, ifc_class="IfcProject", name="Test building")
    run("unit.assign_unit", f, length={"is_metric": True, "raw": unit})
    site = run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    building = run("root.create_entity", f, ifc_class="IfcBuilding", name="Main building")
    storey = run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Level 0")
    run("aggregate.assign_object", f, relating_object=project, products=[site])
    run("aggregate.assign_object", f, relating_object=site, products=[building])
    run("aggregate.assign_object", f, relating_object=building, products=[storey])
    ctx = run("context.add_context", f, context_type="Model")
    body = run(
        "context.add_context",
        f,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
        parent=ctx,
    )

    def place(entity, matrix):
        m = np.array(matrix, dtype=float)
        m[:3, 3] += np.asarray(offset_mm, dtype=float)
        run("geometry.edit_object_placement", f, product=entity, matrix=m)

    def box(entity_class, name, length, height, thickness, pos, rot_z=0.0, material=None):
        e = run("root.create_entity", f, ifc_class=entity_class, name=name)
        run("spatial.assign_container", f, relating_structure=storey, products=[e])
        rep = run(
            "geometry.add_wall_representation",
            f,
            context=body,
            length=length,
            height=height,
            thickness=thickness,
        )
        run("geometry.assign_representation", f, product=e, representation=rep)
        c, s = np.cos(rot_z), np.sin(rot_z)
        m = np.array([[c, -s, 0, pos[0]], [s, c, 0, pos[1]], [0, 0, 1, pos[2]], [0, 0, 0, 1]])
        place(e, m)
        if material:
            mat = run("material.add_material", f, name=material)
            run("material.assign_material", f, products=[e], material=mat)
        return e

    slab = run("root.create_entity", f, ifc_class="IfcSlab", name="Ground floor slab")
    run("spatial.assign_container", f, relating_structure=storey, products=[slab])
    rep = run(
        "geometry.add_slab_representation",
        f,
        context=body,
        depth=200.0,
        polyline=[(0.0, 0.0), (10000.0, 0.0), (10000.0, 10000.0), (0.0, 10000.0)],
    )
    run("geometry.assign_representation", f, product=slab, representation=rep)
    place(slab, np.eye(4))
    run(
        "material.assign_material",
        f,
        products=[slab],
        material=run("material.add_material", f, name="Concrete C30/37"),
    )
    wall_a = box("IfcWall", "Wall South", 10000, 3000, 200, (0, 0, 200), 0.0, "Concrete C30/37")
    box("IfcWall", "Wall East", 10000, 3000, 200, (10200, 0, 200), np.pi / 2, "Concrete C30/37")
    box("IfcWall", "Wall North", 10000, 3000, 200, (10000, 10000, 200), np.pi, "Concrete C30/37")
    box("IfcWall", "Wall West", 10000, 3000, 200, (0, 10000, 200), -np.pi / 2, "Concrete C30/37")
    box("IfcColumn", "Column 1", 400, 3000, 400, (4800, 4800, 200), 0.0, "Steel S355")
    box("IfcDoor", "Entrance door", 900, 2100, 50, (1000, -25, 200), 0.0, "Oak wood")
    box("IfcFurniture", "Office chair", 450, 900, 450, (3000, 3000, 200), 0.0, "Oak wood")
    if pset_lobs:
        pset = run("pset.add_pset", f, product=wall_a, name="Pset_LoBS")
        run("pset.edit_pset", f, pset=pset, properties=pset_lobs)
    f.write(str(path))
    return path
