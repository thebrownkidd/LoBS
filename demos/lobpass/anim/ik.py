"""Two-bone limb IK that returns local joint quaternions, and small aiming helpers."""

from __future__ import annotations

import math

import numpy as np

from . import rig


def cross(a, b):
    """Cross product of two 3-vectors (np.cross is an order of magnitude slower at this size)."""
    return np.array(
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    )


def norm(v) -> float:
    return math.sqrt(float(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def unit(v):
    n = norm(v)
    return v / n if n > 1e-9 else np.zeros_like(v)


def min_reach(l1, l2, max_flex_deg):
    """Closest the end may be to the origin if the mid joint may flex at most ``max_flex_deg`` (0 = straight)."""
    c = math.cos(math.radians(max_flex_deg))
    return math.sqrt(max(l1 * l1 + l2 * l2 + 2 * l1 * l2 * c, 0.0))


def solve_mid(origin, target, l1, l2, pole, max_flex_deg=150.0):
    """Mid joint (knee/elbow) position for a two-bone chain reaching ``target`` (clamped to reach and to the joint's
    flexion limit). The mid joint bends toward ``pole``. Returns (mid, end, reached)."""
    d = target - origin
    dist = norm(d)
    reach = (l1 + l2) * 0.9995
    reached = dist <= reach
    dist_c = min(max(dist, min_reach(l1, l2, max_flex_deg) + 1e-6), reach)
    dirn = unit(d) if dist > 1e-9 else np.array([0.0, 0.0, -1.0])
    a = (l1 * l1 - l2 * l2 + dist_c * dist_c) / (
        2 * dist_c
    )  # distance along dirn to the mid joint's foot point
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    pole_perp = pole - dirn * float(pole @ dirn)
    pole_perp = unit(pole_perp) if norm(pole_perp) > 1e-6 else unit(cross(dirn, [0, 1, 0]))
    mid = origin + dirn * a + pole_perp * h
    return mid, origin + dirn * dist_c, reached


def limb_quats(parent_rot, origin, mid, end, pole):
    """Local quaternions (upper, lower) for a limb whose rest direction is -z and whose hinge is its local y axis.
    ``parent_rot`` is the world rotation of the upper bone's parent. Also returns the upper bone's world rotation."""
    d1, d2 = rig_unit(mid - origin), rig_unit(end - mid)
    # the bend plane is spanned by the pole and the origin->end line (where solve_mid put the mid joint); using the
    # upper bone instead is ill-conditioned for a folded limb, whose upper bone is nearly parallel to the pole
    chord = rig_unit(end - origin)
    pole_perp = unit(np.asarray(pole, float) - chord * float(pole @ chord))
    axis = unit(cross(pole_perp, chord))
    if norm(axis) < 1e-6:
        axis = np.array([0.0, 1.0, 0.0])

    def frame(d):
        z = -d
        x = cross(axis, z)
        x = unit(x) if norm(x) > 1e-9 else np.array([1.0, 0.0, 0.0])
        return np.stack([x, cross(z, x), z], axis=1)

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
        perp = unit(cross(a, [1, 0, 0])) if abs(a[0]) < 0.9 else unit(cross(a, [0, 1, 0]))
        return rig.qaxis(perp, math.pi)
    return rig.qnorm(np.array([1.0 + c, *cross(a, b)]))


def look_angles(head_pos, yaw, target):
    """(yaw, pitch) of ``target`` seen from the head, relative to a body facing ``yaw`` (radians, wrapped)."""
    d = target - head_pos
    rel = (math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
    return rel, math.atan2(d[2], math.hypot(d[0], d[1]))


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)
