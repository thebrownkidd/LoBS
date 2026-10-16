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


_MESH_DIRS = {
    "top": (0, 0, 1), "bottom": (0, 0, -1), "front": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0), "right": (0, -1, 0),
    "+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1),
}  # fmt: skip


def mesh_face_pose(vertices, faces, face, direction=None, offset: float = 0.0):
    """Surface point of a mesh found by casting a ray from its bounding-box centre along a direction
    (a face name or an explicit vector); the anchor's z axis is the outward triangle normal."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if direction is None or (face and face in _MESH_DIRS):
        direction = _MESH_DIRS.get(face or "top")
        if direction is None:
            raise LobsError(
                f"unknown face '{face}' for a mesh (use top/bottom/front/back/left/right, +-x/y/z or direction=)"
            )
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    reach = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) * 1.1 + 1e-6
    origin, ray = centre + d * reach, -d  # start outside, look inward: the first hit faces +d
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    e1, e2 = b - a, c - a
    p = np.cross(ray, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = origin - a
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    w = (q @ ray) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= -1e-9) & (w >= -1e-9) & (u + w <= 1 + 1e-9) & (t > 1e-9)
    if hit.any():
        i = int(np.argmin(np.where(hit, t, np.inf)))
        point = origin + ray * t[i]
    else:  # the centre column misses a concave part: use the triangle extreme along the direction
        cent = (a + b + c) / 3.0
        height = cent @ d
        cand = np.flatnonzero(height >= height.max() - 1e-6)
        lateral = np.linalg.norm(
            (cent[cand] - centre) - np.outer((cent[cand] - centre) @ d, d), axis=1
        )
        i = int(cand[np.argmin(lateral)])
        point = cent[i]
    n = np.cross(e1[i], e2[i])
    n = n / np.linalg.norm(n)
    if n @ d < 0:
        n = -n
    return point + n * offset, m3.mat_to_quat(m3.basis_from_z(n))


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
