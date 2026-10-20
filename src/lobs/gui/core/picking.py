"""Scene queries for the viewport: pick bodies under the cursor, overlay anchor points (no Qt)."""

from __future__ import annotations

import numpy as np

from ... import math3d as m3
from ...viz import drawables
from .camera import OrbitCamera


class ScenePicker:
    """Bounding spheres of every non-plane drawable, tracked against the live state frame."""

    def __init__(self, spec, reg):
        self.reg = reg
        self.items = [d for d in drawables(spec, reg) if d.kind != "plane"]

    def spheres(self, frame):
        centers, radii, bodies = [], [], []
        for it in self.items:
            info = self.reg.bodies[it.body]
            if info.static:
                p, q = info.world_pos, info.world_quat
            else:
                p, q = frame.body_pos[info.dyn], frame.body_quat[info.dyn]
            centers.append(p + m3.quat_rotate(q, it.pos))
            radii.append(max(it.radius, 0.02))
            bodies.append(it.body)
        return centers, radii, bodies

    def pick(
        self, cam: OrbitCamera, px: float, py: float, width: int, height: int, frame
    ) -> int | None:
        """Registry body index under the pixel, or None. Boxes/spheres are hit exactly (slab / quadratic in the
        geom frame); other shapes use a box (cylinder, capsule) or bounding sphere (mesh)."""
        o, d = cam.ray(px, py, width, height)
        best, best_t = None, np.inf
        for it in self.items:
            info = self.reg.bodies[it.body]
            if info.static:
                p, q = info.world_pos, info.world_quat
            else:
                p, q = frame.body_pos[info.dyn], frame.body_quat[info.dyn]
            c = p + m3.quat_rotate(q, it.pos)
            qg = m3.quat_mul(q, it.quat)
            lo_, ld = m3.quat_rotate(m3.quat_conj(qg), o - c), m3.quat_rotate(m3.quat_conj(qg), d)
            t = _hit(it, lo_, ld)
            if t is not None and t < best_t:
                best, best_t = it.body, t
        return best

    def body_path(self, body: int) -> str:
        return self.reg.bodies[body].path

    def actor_of(self, body: int) -> str:
        return self.reg.bodies[body].path.split("/")[0]

    def body_position(self, frame, body: int) -> np.ndarray:
        info = self.reg.bodies[body]
        return np.asarray(info.world_pos if info.static else frame.body_pos[info.dyn], dtype=float)


def _hit(it, o, d):
    """Ray parameter of the first hit on a drawable in its own frame, or None."""
    if it.kind == "sphere":
        return _ray_sphere(o, d, it.size[0])
    if it.kind == "box":
        half = np.asarray(it.size, dtype=float) / 2.0
    elif it.kind == "cylinder":
        half = np.array([it.size[0], it.size[0], it.size[1] / 2.0])
    elif it.kind == "capsule":
        half = np.array([it.size[0], it.size[0], it.size[1] / 2.0 + it.size[0]])
    else:
        return _ray_sphere(o, d, max(it.radius, 0.02))
    t0, t1 = 0.0, np.inf
    for k in range(3):
        if abs(d[k]) < 1e-12:
            if abs(o[k]) > half[k]:
                return None
            continue
        a, b = (-half[k] - o[k]) / d[k], (half[k] - o[k]) / d[k]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return None
    return t0


def _ray_sphere(o, d, r):
    b = float(np.dot(o, d))
    disc = b * b - (float(np.dot(o, o)) - r * r)
    if disc < 0:
        return None
    t = -b - np.sqrt(disc)
    return t if t >= 0 else None
