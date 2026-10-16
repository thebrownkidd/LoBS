"""Normalize a SceneIR to meters / Z-up / a sane origin (plan 11.4)."""

from __future__ import annotations

import numpy as np

from ..errors import ImportFailure
from .decisions import DecisionLog
from .ir import SceneIR
from .readers.mesh import meters_per_unit

_RX90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)  # +90 deg about X: Y-up -> Z-up
HUGE_COORD = 1.0e4  # metres: beyond this, double-precision-friendly recentring kicks in


def _scale_ir(ir: SceneIR, s: float) -> None:
    for n in ir.nodes.values():
        n.local = n.local.copy()
        n.local[:3, 3] *= s
        for g in n.geoms:
            g.transform = g.transform.copy()
            g.transform[:3, 3] *= s
            if g.vertices is not None:
                g.vertices = g.vertices * s
            if g.size is not None:
                g.size = tuple(x * s for x in g.size)
            if g.radius is not None:
                g.radius *= s
            if g.length is not None:
                g.length *= s
        if n.physics and n.physics.com is not None:
            n.physics.com = tuple(x * s for x in n.physics.com)
    for j in ir.joints:
        j.pos = tuple(x * s for x in j.pos)
        if j.type == "prismatic" and j.limits is not None:
            j.limits = tuple(x * s for x in j.limits)


def _transform_roots(ir: SceneIR, rot: np.ndarray, shift: np.ndarray) -> None:
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = shift
    for n in ir.roots():
        n.local = t @ n.local
    # root-less quantities that live in world axes
    if ir.world_physics and ir.world_physics.gravity is not None:
        ir.world_physics.gravity = tuple(rot @ np.asarray(ir.world_physics.gravity))


def _world_extent(ir: SceneIR):
    world = ir.world_transforms()
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for nid, n in ir.nodes.items():
        for g in n.geoms:
            if g.vertices is not None and len(g.vertices):
                pts = (world[nid] @ g.transform) @ np.c_[g.vertices, np.ones(len(g.vertices))].T
                pts = pts[:3].T
            else:
                pts = (world[nid] @ g.transform)[:3, 3][None]
            lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
    if not np.all(np.isfinite(lo)):
        raise ImportFailure("the source contains no geometry")
    return lo, hi


def normalize(
    ir: SceneIR, log: DecisionLog, *, units="auto", up_axis="auto", origin="auto", scale=1.0
):
    """Mutates ``ir`` in place. Returns the georeference dict to store (or None)."""
    # ---- units
    mpu = meters_per_unit(units) if units not in (None, "auto") else None
    if mpu is not None:
        log.add("scene", "units", f"{mpu} m/unit", "override", rationale="units given explicitly")
    elif ir.source.meters_per_unit is not None:
        mpu = ir.source.meters_per_unit
        log.add(
            "scene",
            "units",
            f"{mpu} m/unit",
            "source",
            rationale="units declared by the source file",
        )
    else:
        mpu = 1.0
        lo, hi = _world_extent(ir)
        ext = float(np.max(hi - lo))
        flagged = ext > 1000.0
        log.add("scene", "units", "1.0 m/unit", "default", "low" if flagged else "medium",
                f"the format stores no units; assumed metres (largest extent {ext:.3g})"
                + (" - looks like millimetres, pass units='mm'" if flagged else ""), review=True)  # fmt: skip
    s = mpu * float(scale)
    if s != 1.0:
        _scale_ir(ir, s)
    # ---- up axis
    up = ir.source.up_axis if up_axis in (None, "auto") else str(up_axis).lower()
    if up_axis not in (None, "auto"):
        log.add("scene", "up_axis", up, "override", rationale="up axis given explicitly")
    elif up is None:
        up = "z"
        log.add(
            "scene",
            "up_axis",
            "z",
            "default",
            "medium",
            "format does not declare an up axis; assumed Z",
            review=True,
        )
    else:
        log.add("scene", "up_axis", up, "source", rationale="declared by the source file")
    rot = np.eye(3)
    if up == "y":
        rot = _RX90
    elif up != "z":
        raise ImportFailure(f"unsupported up axis '{up}'")
    shift = np.zeros(3)
    _transform_roots(ir, rot, shift)
    # ---- origin
    lo, hi = _world_extent(ir)
    georef = ir.source.georef
    offset = np.zeros(3)
    if isinstance(origin, (list, tuple, np.ndarray)):
        offset = np.asarray(origin, dtype=float)
        log.add("scene", "origin", offset.tolist(), "override", rationale="origin given explicitly")
    elif origin == "recenter" or (origin == "auto" and np.max(np.abs(np.r_[lo, hi])) > HUGE_COORD):
        offset = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        log.add("scene", "origin", offset.tolist(), "heuristic" if origin == "auto" else "override",
                rationale="coordinates are huge (map/CRS scale): recentred on the scene footprint, min z = 0")  # fmt: skip
    if np.any(offset):
        _transform_roots(ir, np.eye(3), -offset)
        base = (
            np.asarray(georef["offset"], float) * (1.0 if georef.get("meters") else s)
            if georef and georef.get("offset") is not None
            else np.zeros(3)
        )
        georef = {"crs": (georef or {}).get("crs"), "offset_m": (base + offset).tolist()}
    elif georef:
        georef = {
            "crs": georef.get("crs"),
            "offset_m": (
                np.asarray(georef.get("offset", (0, 0, 0)), float)
                * (1.0 if georef.get("meters") else s)
            ).tolist(),
        }
    ir.source.meters_per_unit = 1.0
    ir.source.up_axis = "z"
    return georef
