"""Collision representation for imported geometry: primitives, OBB boxes, convex hulls, CoACD sets,
and mass properties from meshes / primitives (plan 11.5)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import inertia as I
from .. import math3d as m3
from ..schema import Box, Capsule, Cylinder, Sphere

STRATEGIES = ("auto", "primitive", "box_fit", "convex_hull", "decompose", "mesh", "none")


@dataclass
class FitShape:
    kind: str  # box | hull | mesh
    strategy: str
    center: np.ndarray | None = None  # box: centre in the geom frame
    rot: np.ndarray | None = None  # box: 3x3
    size: np.ndarray | None = None  # box: full extents
    vertices: np.ndarray | None = None  # hull / mesh
    faces: np.ndarray | None = None


@dataclass
class FitResult:
    shapes: list = field(default_factory=list)
    strategy: str = "none"
    confidence: str = "high"
    note: str = ""
    review: bool = False


@dataclass
class MeshProps:
    watertight: bool
    volume: float  # of the solid used for mass (mesh if watertight, else its convex hull)
    centroid: np.ndarray
    inertia_unit: np.ndarray  # about centroid, density 1
    mesh_volume: float
    hull_volume: float
    obb_center: np.ndarray
    obb_rot: np.ndarray
    obb_extents: np.ndarray
    exact: bool  # mass props come from the real mesh (watertight), not the hull


def _tm(v, f):
    import trimesh

    return trimesh.Trimesh(vertices=np.asarray(v, float), faces=np.asarray(f, int), process=True)


def mesh_props(v, f) -> MeshProps:
    mesh = _tm(v, f)
    hull = mesh.convex_hull
    obb = mesh.bounding_box_oriented
    t = np.asarray(obb.primitive.transform)
    watertight = bool(mesh.is_watertight and mesh.volume > 1e-12)
    src = mesh if watertight else hull
    if src.volume > 1e-12:
        centroid, inertia = np.asarray(src.center_mass), np.asarray(src.moment_inertia)
        volume = abs(float(src.volume))
    else:  # degenerate / flat: treat as a thin box of its OBB
        ext = np.maximum(np.asarray(obb.primitive.extents), 1e-3)
        volume = float(np.prod(ext))
        centroid = t[:3, 3]
        inertia = I.shape_inertia(Box(size=tuple(ext)), volume)
        inertia = t[:3, :3] @ inertia @ t[:3, :3].T
    return MeshProps(watertight, volume, centroid, inertia, abs(float(mesh.volume)) if mesh.is_watertight else 0.0,
                     abs(float(hull.volume)), t[:3, 3], t[:3, :3], np.asarray(obb.primitive.extents), watertight)  # fmt: skip


def _hull(v, f, max_faces):
    mesh = _tm(v, f)
    hull = mesh.convex_hull
    if len(hull.faces) > max_faces:
        try:
            hull = hull.simplify_quadric_decimation(face_count=max_faces).convex_hull
        except Exception:  # noqa: BLE001 - decimation is best effort
            pass
    return np.asarray(hull.vertices), np.asarray(hull.faces)


def _decompose(v, f, params, max_hulls):
    import coacd

    coacd.set_log_level("error")
    cp = dict(params.get("coacd") or {})
    mesh = coacd.Mesh(np.asarray(v, float), np.asarray(f, int))
    parts = coacd.run_coacd(mesh, threshold=params["decompose_threshold"], seed=0,
                            max_convex_hull=int(max_hulls), **cp)  # fmt: skip
    return [(np.asarray(pv), np.asarray(pf)) for pv, pf in parts]


def fit_collision(
    v, f, strategy: str, params: dict, fixed: bool, props: MeshProps | None = None
) -> FitResult:
    """Choose collision shapes for one mesh. ``strategy`` is a profile/override value."""
    props = props or mesh_props(v, f)
    max_faces = int(params["max_faces"])
    max_hulls = int(params["max_hulls_fixed" if fixed else "max_hulls_dynamic"])

    def box() -> FitShape:
        return FitShape(
            "box",
            "box_fit",
            center=props.obb_center,
            rot=props.obb_rot,
            size=np.maximum(props.obb_extents, 1e-4),
        )

    def hull() -> FitShape:
        hv, hf = _hull(v, f, max_faces)
        return FitShape("hull", "convex_hull", vertices=hv, faces=hf)

    if strategy == "none":
        return FitResult([], "none", "high", "collision disabled")
    if strategy == "box_fit":
        return FitResult([box()], "box_fit", "high", "oriented bounding box (requested)")
    if strategy == "convex_hull":
        return FitResult([hull()], "convex_hull", "high", "convex hull (requested)")
    if strategy == "mesh":
        if not fixed:
            return FitResult(
                [hull()],
                "convex_hull",
                "medium",
                "triangle mesh is static-only; used a convex hull",
                review=True,
            )
        return FitResult([FitShape("mesh", "mesh", vertices=np.asarray(v), faces=np.asarray(f))], "mesh", "high",
                         "exact triangle mesh (static; Chrono only)")  # fmt: skip
    if strategy == "decompose":
        return _decompose_result(v, f, params, max_hulls, hull)
    # ---- auto
    obb_vol = float(np.prod(np.maximum(props.obb_extents, 1e-9)))
    solid = props.mesh_volume if props.watertight else props.hull_volume
    if props.watertight or props.hull_volume > 0:
        if solid / obb_vol >= params["box_fit_ratio"]:
            return FitResult(
                [box()], "box_fit", "high", f"mesh fills {solid / obb_vol:.0%} of its oriented box"
            )
        if (
            props.watertight
            and props.mesh_volume / max(props.hull_volume, 1e-12) >= params["hull_ratio"]
        ):
            return FitResult([hull()], "convex_hull", "high",
                             f"mesh fills {props.mesh_volume / props.hull_volume:.0%} of its convex hull")  # fmt: skip
    if not props.watertight:
        res = _decompose_result(v, f, params, max_hulls, hull)
        res.confidence, res.review = "low", True
        res.note = "mesh is not watertight; " + res.note
        return res
    return _decompose_result(v, f, params, max_hulls, hull)


def _decompose_result(v, f, params, max_hulls, hull_fn) -> FitResult:
    try:
        parts = _decompose(v, f, params, max_hulls)
    except ImportError:
        return FitResult([hull_fn()], "convex_hull", "low", "coacd not installed: used a single convex hull (concavities filled)", review=True)  # fmt: skip
    shapes = [FitShape("hull", "decompose", vertices=pv, faces=pf) for pv, pf in parts]
    return FitResult(
        shapes,
        "decompose",
        "medium",
        f"approximate convex decomposition into {len(shapes)} hulls (CoACD, seed 0)",
    )


# ---------------------------------------------------------------- mass properties
def primitive_props(kind: str, size=None, radius=None, length=None):
    """(volume, inertia about own centre for density 1)."""
    shape = {
        "box": lambda: Box(size=tuple(size)), "sphere": lambda: Sphere(radius=radius),
        "cylinder": lambda: Cylinder(radius=radius, length=length), "capsule": lambda: Capsule(radius=radius, length=length),
    }[kind]()  # fmt: skip
    vol = I.shape_volume(shape)
    return vol, I.shape_inertia(shape, vol)


def combine_parts(parts):
    """parts: [(mass, com, inertia_about_com)] -> (mass, com, inertia about com)."""
    return I.combine(parts)


def pose_from_matrix(t: np.ndarray):
    return t[:3, 3].copy(), m3.mat_to_quat(t[:3, :3])
