"""Mass properties: primitive closed forms, parallel-axis combination, principal axes.

Everything is expressed in the body frame. Inertia matrices are about the relevant COM.
"""

from __future__ import annotations

import numpy as np

from . import math3d as m3
from .errors import LobsError
from .schema import Box, Capsule, Cylinder, Sphere
from .schema.actor import BodySpec

MASSLESS_SHAPES = ("plane", "triangle_mesh", "heightfield")


def shape_volume(shape) -> float:
    if isinstance(shape, Box):
        return float(np.prod(shape.size))
    if isinstance(shape, Sphere):
        return 4.0 / 3.0 * np.pi * shape.radius**3
    if isinstance(shape, Cylinder):
        return np.pi * shape.radius**2 * shape.length
    if isinstance(shape, Capsule):
        return np.pi * shape.radius**2 * shape.length + 4.0 / 3.0 * np.pi * shape.radius**3
    raise LobsError(
        f"cannot compute volume of shape type '{shape.type}' (give explicit mass props)"
    )


def shape_inertia(shape, mass: float) -> np.ndarray:
    """3x3 inertia about the shape's own center for the given mass (axes = shape axes)."""
    if isinstance(shape, Box):
        a, b, c = shape.size
        return np.diag(
            [mass / 12 * (b * b + c * c), mass / 12 * (a * a + c * c), mass / 12 * (a * a + b * b)]
        )
    if isinstance(shape, Sphere):
        return np.eye(3) * (0.4 * mass * shape.radius**2)
    if isinstance(shape, Cylinder):
        r, h = shape.radius, shape.length
        ixx = mass * (3 * r * r + h * h) / 12
        return np.diag([ixx, ixx, 0.5 * mass * r * r])
    if isinstance(shape, Capsule):
        r, length = shape.radius, shape.length
        vc = np.pi * r * r * length
        vs = 4.0 / 3.0 * np.pi * r**3
        mc = mass * vc / (vc + vs)
        mh = mass * vs / (vc + vs) / 2.0  # one hemisphere
        izz = 0.5 * mc * r * r + 2 * (0.4 * mh * r * r)
        # hemisphere: about face-centre axis 2/5 m r^2; to its COM subtract m (3r/8)^2; then shift
        ixx_h = mh * (0.4 * r * r - (3 * r / 8) ** 2) + mh * (length / 2 + 3 * r / 8) ** 2
        ixx = mc * (r * r / 4 + length * length / 12) + 2 * ixx_h
        return np.diag([ixx, ixx, izz])
    raise LobsError(f"cannot compute inertia of shape type '{shape.type}'")


def parallel_axis(inertia: np.ndarray, mass: float, offset) -> np.ndarray:
    d = np.asarray(offset, dtype=float)
    return inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def combine(parts) -> tuple[float, np.ndarray, np.ndarray]:
    """Combine (mass, com, inertia_about_own_com) parts (same frame) -> (mass, com, I about com)."""
    parts = list(parts)
    total = sum(p[0] for p in parts)
    if total <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    com = sum(p[0] * np.asarray(p[1], dtype=float) for p in parts) / total
    inertia = sum(parallel_axis(p[2], p[0], np.asarray(p[1]) - com) for p in parts)
    return float(total), com, inertia


def principal_axes(inertia: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(diag, quat) with inertia = R diag R^T, R = rotation of the principal frame in body axes."""
    w, v = np.linalg.eigh(0.5 * (inertia + inertia.T))
    if np.linalg.det(v) < 0:
        v[:, 2] = -v[:, 2]
    return w, m3.mat_to_quat(v)


def resolve_body_inertial(body: BodySpec, default_density: float = 1000.0):
    """Resolve (mass, com, inertia 3x3 about com) in the body frame from a BodySpec."""
    spec = body.inertial
    if spec.mode == "explicit":
        if spec.mass is None or spec.inertia is None:
            raise LobsError(f"body '{body.name}': explicit inertial needs mass and inertia")
        return (
            float(spec.mass),
            np.asarray(spec.com, dtype=float),
            np.asarray(spec.inertia, dtype=float),
        )

    geoms = [g for g in body.geoms if g.role in ("both", "collision")]
    sized = []
    for g in geoms:
        if g.shape.type in MASSLESS_SHAPES:
            continue  # static-only shapes carry no mass
        sized.append((g, shape_volume(g.shape)))
    if not sized:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    densities = []
    for g, _ in sized:
        d = spec.density
        if d is None and g.material is not None:
            d = g.material.density
        densities.append(d if d is not None else default_density)
    masses = [v * d for (_, v), d in zip(sized, densities, strict=True)]
    if spec.mass is not None:
        total = sum(masses)
        scale = spec.mass / total if total > 0 else 0.0
        masses = [m * scale for m in masses]
    parts = []
    for (g, _), m in zip(sized, masses, strict=True):
        r = m3.quat_to_mat(g.pose.quat)
        parts.append((m, np.asarray(g.pose.pos), r @ shape_inertia(g.shape, m) @ r.T))
    return combine(parts)
