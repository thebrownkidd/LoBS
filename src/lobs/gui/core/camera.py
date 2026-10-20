"""Orbit camera that mirrors MuJoCo's free camera, with projection and picking rays (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FOVY_DEG = 45.0  # MuJoCo's default vertical field of view


@dataclass
class OrbitCamera:
    lookat: np.ndarray = field(default_factory=lambda: np.zeros(3))
    distance: float = 5.0
    azimuth: float = 135.0  # degrees
    elevation: float = -25.0  # degrees (negative looks down)

    def as_dict(self) -> dict:
        return {
            "lookat": np.asarray(self.lookat, dtype=float),
            "distance": float(self.distance),
            "azimuth": float(self.azimuth),
            "elevation": float(self.elevation),
        }

    # ---- geometry
    def forward(self) -> np.ndarray:
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    def position(self) -> np.ndarray:
        return np.asarray(self.lookat, dtype=float) - self.distance * self.forward()

    def basis(self):
        """(right, up, forward) unit vectors of the camera in world coordinates."""
        f = self.forward()
        right = np.cross(f, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        return right, up, f

    def project(self, points, width: int, height: int):
        """Pixels (n, 2) and depths (n,) for world points; depth <= 0 is behind the camera."""
        p = np.atleast_2d(np.asarray(points, dtype=float)) - self.position()
        right, up, f = self.basis()
        x, y, z = p @ right, p @ up, p @ f
        focal = 0.5 * height / np.tan(np.radians(FOVY_DEG) / 2.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            px = width / 2.0 + focal * x / z
            py = height / 2.0 - focal * y / z
        return np.stack([px, py], axis=1), z

    def ray(self, px: float, py: float, width: int, height: int):
        """World-space ray (origin, unit direction) through a pixel."""
        focal = 0.5 * height / np.tan(np.radians(FOVY_DEG) / 2.0)
        right, up, f = self.basis()
        d = f + right * (px - width / 2.0) / focal - up * (py - height / 2.0) / focal
        return self.position(), d / np.linalg.norm(d)

    # ---- interaction
    def orbit(self, dx: float, dy: float, speed: float = 0.4) -> None:
        self.azimuth = (self.azimuth - dx * speed) % 360.0
        self.elevation = float(np.clip(self.elevation - dy * speed, -89.0, 89.0))

    def pan(self, dx: float, dy: float, height: int) -> None:
        right, up, _ = self.basis()
        scale = 2.0 * self.distance * np.tan(np.radians(FOVY_DEG) / 2.0) / max(height, 1)
        self.lookat = np.asarray(self.lookat, dtype=float) + (-dx * right + dy * up) * scale

    def zoom(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.9**steps), 0.05, 1e4))

    def frame(self, lo, hi) -> None:
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        self.lookat = (lo + hi) / 2.0
        self.distance = max(1.2 * float(np.linalg.norm(hi - lo)), 1.0)


def pick_sphere(origin, direction, centers, radii):
    """Index of the nearest sphere hit by the ray, or None."""
    best, best_t = None, np.inf
    o = np.asarray(origin, dtype=float)
    for i, (c, r) in enumerate(zip(centers, radii, strict=True)):
        oc = o - np.asarray(c, dtype=float)
        b = float(np.dot(oc, direction))
        disc = b * b - (float(np.dot(oc, oc)) - r * r)
        if disc < 0:
            continue
        t = -b - np.sqrt(disc)
        if t < 0:
            t = -b + np.sqrt(disc)
        if 0 <= t < best_t:
            best, best_t = i, t
    return best
