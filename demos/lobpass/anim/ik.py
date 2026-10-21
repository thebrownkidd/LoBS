"""Two-bone limb IK that returns local joint quaternions, and small aiming helpers."""

from __future__ import annotations

import math

import numpy as np

from . import rig


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros_like(v)


def solve_mid(origin, target, l1, l2, pole):
    """Mid joint (knee/elbow) position for a two-bone chain reaching ``target`` (clamped to reach). The mid joint
    bends toward ``pole``. Returns (mid, end, reached)."""
    d = target - origin
    dist = float(np.linalg.norm(d))
    reach = (l1 + l2) * 0.9995
    reached = dist <= reach
    dist_c = min(max(dist, abs(l1 - l2) + 1e-6), reach)
    dirn = unit(d) if dist > 1e-9 else np.array([0.0, 0.0, -1.0])
    a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (2 * dist_c)  # distance along dirn to the mid joint's foot point
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pole_perp = pole - dirn * float(np.dot(pole, dirn))
    pole_perp = unit(pole_perp) if np.linalg.norm(pole_perp) > 1e-6 else unit(np.cross(dirn, [0, 1, 0]))
    mid = origin + dirn * a + pole_perp * h
    return mid, origin + dirn * dist_c, reached


def limb_quats(parent_rot, origin, mid, end, pole):
    """Local quaternions (upper, lower) for a limb whose rest direction is -z and whose hinge is its local y axis.
    ``parent_rot`` is the world rotation of the upper bone's parent. Also returns the upper bone's world rotation."""
    d1, d2 = rig_unit(mid - origin), rig_unit(end - mid)
    pole_perp = unit(np.asarray(pole, float) - d1 * float(np.dot(pole, d1)))
    axis = unit(np.cross(pole_perp, d1))
    if np.linalg.norm(axis) < 1e-6:
        axis = np.array([0.0, 1.0, 0.0])

    def frame(d):
        z = -d
        x = np.cross(axis, z)
        x = unit(x) if np.linalg.norm(x) > 1e-9 else np.array([1.0, 0.0, 0.0])
        return np.stack([x, np.cross(z, x), z], axis=1)

    r1, r2 = frame(d1), frame(d2)
    return rig.mat_to_q(parent_rot.T @ r1), rig.mat_to_q(r1.T @ r2), r1


def rig_unit(v):
    return unit(np.asarray(v, float))


def aim_quat(from_dir, to_dir):
    """Shortest rotation taking unit vector from_dir to to_dir."""
    a, b = unit(np.asarray(from_dir, float)), unit(np.asarray(to_dir, float))
    c = float(np.dot(a, b))
    if c > 0.999999:
        return rig.IDENT.copy()
    if c < -0.999999:
        perp = unit(np.cross(a, [1, 0, 0])) if abs(a[0]) < 0.9 else unit(np.cross(a, [0, 1, 0]))
        return rig.qaxis(perp, math.pi)
    return rig.qnorm(np.array([1.0 + c, *np.cross(a, b)]))


def look_angles(head_pos, yaw, target):
    """(yaw, pitch) of ``target`` seen from the head, relative to a body facing ``yaw`` (radians, wrapped)."""
    d = target - head_pos
    rel = (math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    return rel, math.atan2(d[2], math.hypot(d[0], d[1]))


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)
