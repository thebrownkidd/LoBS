"""OBJ / STL / PLY via trimesh. Units are NOT in these files: pass ``units=`` or accept the flag."""

from __future__ import annotations

import numpy as np

from ..ir import SceneIR, SourceInfo, bake_mesh_node

UNIT_TO_M = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3,
    "in": 0.0254, "inch": 0.0254, "ft": 0.3048,
}  # fmt: skip


def meters_per_unit(units) -> float | None:
    if units is None or units == "auto":
        return None
    if isinstance(units, (int, float)):
        return float(units)
    if units not in UNIT_TO_M:
        raise ValueError(f"unknown units '{units}' (known: {sorted(UNIT_TO_M)})")
    return UNIT_TO_M[units]


def read(path, *, units=None, up_axis="z", split=True, **_):
    import trimesh

    loaded = trimesh.load(str(path), force="scene")
    src = SourceInfo(
        path=str(path), format="mesh", meters_per_unit=meters_per_unit(units), up_axis=up_axis
    )
    ir = SceneIR(src)
    if src.meters_per_unit is None:
        src.notes.append("units are not stored in this format; unit heuristics will be applied")
    for k, name in enumerate(sorted(loaded.graph.nodes_geometry)):
        transform, gname = loaded.graph[name]
        mesh = loaded.geometry[gname]
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            ir.warnings.append(f"skipped non-mesh geometry '{gname}'")
            continue
        v = trimesh.transform_points(mesh.vertices, transform)
        parts = [(v, mesh.faces)]
        if split:  # these formats carry no structure: disconnected solids are separate elements
            pieces = trimesh.Trimesh(vertices=v, faces=mesh.faces, process=False).split(
                only_watertight=False
            )
            if len(pieces) > 1:
                parts = [(np.asarray(p.vertices), np.asarray(p.faces)) for p in pieces]
        for i, (pv, pf) in enumerate(parts):
            suffix = f".{i}" if len(parts) > 1 else ""
            ir.add(bake_mesh_node(f"{path.stem}/{gname}#{k}{suffix}", f"{gname}{suffix}", pv, pf))
    if not ir.nodes:
        ir.warnings.append("no mesh geometry found")
    return ir
