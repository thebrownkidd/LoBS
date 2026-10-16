"""Spatial math. Quaternions are (w, x, y, z), Hamilton product, unit length."""

from __future__ import annotations

import numpy as np

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
_TINY = 1e-12


def v3(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {a.shape}")
    return a


def quat_normalize(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / n


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a, b) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_mat(q) -> np.ndarray:
    """Rotation matrix (..., 3, 3) of unit quaternion(s)."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def mat_to_quat(r) -> np.ndarray:
    """Quaternion of a single 3x3 rotation matrix (Shepperd's method), w >= 0."""
    r = np.asarray(r, dtype=float)
    t = np.trace(r)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
    elif r[0, 0] >= r[1, 1] and r[0, 0] >= r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        q = [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
    elif r[1, 1] >= r[2, 2]:
        s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        q = [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        q = [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
    q = quat_normalize(np.array(q))
    return -q if q[0] < 0 else q


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector(s) v (..., 3) by quaternion(s) q (..., 4)."""
    return np.einsum("...ij,...j->...i", quat_to_mat(q), np.asarray(v, dtype=float))


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = v3(axis)
    n = np.linalg.norm(axis)
    if n < _TINY:
        raise ValueError("zero rotation axis")
    h = 0.5 * angle
    return np.concatenate([[np.cos(h)], np.sin(h) * axis / n])


def quat_exp(rotvec) -> np.ndarray:
    """Quaternion for rotation vector(s) (..., 3): axis * angle."""
    rv = np.asarray(rotvec, dtype=float)
    ang = np.linalg.norm(rv, axis=-1, keepdims=True)
    half = 0.5 * ang
    # sin(half)/ang -> 0.5 as ang -> 0
    safe = np.where(ang < 1e-12, 1.0, ang)
    k = np.where(ang < 1e-12, 0.5, np.sin(half) / safe)
    return np.concatenate([np.cos(half), rv * k], axis=-1)


def quat_log(q) -> np.ndarray:
    """Rotation vector of a single unit quaternion (shortest representation)."""
    q = quat_normalize(q)
    if q[0] < 0:
        q = -q
    vec = q[1:]
    n = np.linalg.norm(vec)
    if n < 1e-14:
        return 2.0 * vec
    return vec / n * (2.0 * np.arctan2(n, q[0]))


def quat_angle(q) -> float:
    """Geodesic rotation angle in [0, pi] (double-cover aware)."""
    q = np.asarray(q, dtype=float)
    return float(2.0 * np.arctan2(np.linalg.norm(q[1:]), abs(q[0])))


def quat_same_rotation(a, b, tol: float = 1e-9) -> bool:
    return quat_angle(quat_mul(quat_conj(a), b)) < tol


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    return a - 2.0 * np.pi * np.ceil((a - np.pi) / (2.0 * np.pi))


def angle_between(a, b) -> float:
    """Unsigned angle in [0, pi] via atan2; NaN if either vector is (near) zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.linalg.norm(a) < _TINY or np.linalg.norm(b) < _TINY:
        return float("nan")
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)))


def basis_from_z(n) -> np.ndarray:
    """Rotation matrix whose z column is the unit vector n (x chosen deterministically)."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n], axis=1)


def pose_compose(p1, q1, p2, q2):
    """(p1, q1) o (p2, q2)."""
    return np.asarray(p1) + quat_rotate(q1, p2), quat_mul(q1, q2)


def pose_inverse(p, q):
    qi = quat_conj(q)
    return -quat_rotate(qi, p), qi
