"""Face anchor placement: pose of a point on a primitive's surface; z axis = outward normal."""

from __future__ import annotations

import numpy as np

from . import math3d as m3
from .errors import LobsError
from .schema import Box, Capsule, Cylinder, Sphere

_BOX_ALIASES = {
    "top": "+z", "bottom": "-z", "front": "+x", "back": "-x", "left": "+y", "right": "-y",
}  # fmt: skip
_AXIS = {"x": 0, "y": 1, "z": 2}


def canonical_box_face(face: str) -> str:
    return _BOX_ALIASES.get(face, face)


def _unit(sign: str, axis: str) -> np.ndarray:
    n = np.zeros(3)
    n[_AXIS[axis]] = 1.0 if sign == "+" else -1.0
    return n


def face_pose(shape, face: str | None, uv=(0.0, 0.0), offset: float = 0.0, angle: float = 0.0,
              direction=(0.0, 0.0, 1.0)):  # fmt: skip
    """Return (pos, quat) in the shape's frame."""
    if isinstance(shape, Box):
        f = canonical_box_face(face or "")
        if len(f) != 2 or f[0] not in "+-" or f[1] not in _AXIS:
            raise LobsError(f"unknown box face '{face}' (use +x,-x,+y,-y,+z,-z or top/bottom/...)")
        n = _unit(f[0], f[1])
        half = np.asarray(shape.size) / 2.0
        r = m3.basis_from_z(n)
        t1, t2 = r[:, 0], r[:, 1]
        pos = (
            n * (abs(n) @ half + offset)
            + uv[0] * (abs(t1) @ half) * t1
            + uv[1] * (abs(t2) @ half) * t2
        )
        return pos, m3.mat_to_quat(r)
    if isinstance(shape, (Cylinder, Capsule)):
        half_len = shape.length / 2.0
        cap = shape.radius if isinstance(shape, Capsule) else 0.0
        if face in ("top", "bottom"):
            s = 1.0 if face == "top" else -1.0
            n = np.array([0.0, 0.0, s])
            r = m3.basis_from_z(n)
            t1, t2 = r[:, 0], r[:, 1]
            pos = n * (half_len + cap + offset) + shape.radius * (uv[0] * t1 + uv[1] * t2)
            if isinstance(shape, Capsule):
                pos = n * (half_len + cap + offset)  # cap apex; uv ignored
            return pos, m3.mat_to_quat(r)
        if face == "side":
            n = np.array([np.cos(angle), np.sin(angle), 0.0])
            pos = n * (shape.radius + offset) + np.array([0.0, 0.0, uv[1] * half_len])
            return pos, m3.mat_to_quat(m3.basis_from_z(n))
        raise LobsError(f"unknown face '{face}' for {shape.type} (use top, bottom or side)")
    if isinstance(shape, Sphere):
        n = np.asarray(direction, dtype=float)
        if np.linalg.norm(n) < 1e-12:
            raise LobsError("sphere face needs a non-zero direction")
        n = n / np.linalg.norm(n)
        return n * (shape.radius + offset), m3.mat_to_quat(m3.basis_from_z(n))
    raise LobsError(f"anchor_on_face is not supported for shape type '{shape.type}' yet")
