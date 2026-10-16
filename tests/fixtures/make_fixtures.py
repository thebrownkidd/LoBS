"""Generate small foreign-format test worlds at test time (no binary blobs, no licensing issues)."""

from __future__ import annotations

from pathlib import Path

import trimesh


def _box(extents, center):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(center)
    return m


def make_gltf_room(path: Path, extras=None) -> Path:
    """glTF is Y-up. Floor slab top at y=0; a 0.4 m crate on it; a wall; an L-shaped (non-convex) bracket."""
    scene = trimesh.Scene()
    floor = _box((10, 0.2, 10), (0, -0.1, 0))
    crate = _box((0.4, 0.4, 0.4), (1.0, 0.2, 0.5))
    wall = _box((10, 3, 0.2), (0, 1.5, -5))
    for name, geom in (("floor", floor), ("crate", crate), ("north_wall", wall)):
        if extras and name in extras:
            geom.metadata["gltf_extras"] = extras[name]
        scene.add_geometry(geom, node_name=name, geom_name=f"{name}_mesh")
    a = _box((1.0, 0.2, 0.2), (-2.0, 0.1, 1.0))
    b = _box((0.2, 0.2, 1.0), (-2.4, 0.1, 1.4))
    bracket = trimesh.util.concatenate([a, b])
    scene.add_geometry(bracket, node_name="bracket", geom_name="bracket_mesh")
    path.write_bytes(scene.export(file_type="glb"))
    return path


def make_obj_mm(path: Path) -> Path:
    """A 400 x 400 x 400 'mm' cube sitting on a 4000 x 4000 x 100 mm slab (units absent from OBJ)."""
    slab = _box((4000, 4000, 100), (0, 0, -50))
    cube = _box((400, 400, 400), (0, 0, 200))
    scene = trimesh.Scene()
    scene.add_geometry(slab, node_name="slab", geom_name="slab")
    scene.add_geometry(cube, node_name="cube", geom_name="cube")
    path.write_text(scene.export(file_type="obj"))
    return path
