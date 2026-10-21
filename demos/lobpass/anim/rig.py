"""Skeleton (B2): 21 joints, proportions in fractions of the player's height H, plus quaternion helpers and
forward kinematics.

Frames: world is Z-up. A joint's local frame has x forward, y left, z up in the rest pose (arms hanging, standing
straight). Quaternions are (w, x, y, z). The pelvis quaternion is a *world* rotation; every other joint's is local
to its parent. Rest offsets are measured from the parent joint."""

from __future__ import annotations

import math

import numpy as np

from .. import constants as C

JOINTS = (
    "pelvis", "spine", "chest", "neck", "head",
    "clavicle_L", "clavicle_R", "upper_arm_L", "upper_arm_R", "forearm_L", "forearm_R", "hand_L", "hand_R",
    "thigh_L", "thigh_R", "shin_L", "shin_R", "foot_L", "foot_R", "toe_L", "toe_R",
)  # fmt: skip
J = {n: i for i, n in enumerate(JOINTS)}
PARENT = {
    "spine": "pelvis", "chest": "spine", "neck": "chest", "head": "neck",
    "clavicle_L": "chest", "clavicle_R": "chest", "upper_arm_L": "clavicle_L", "upper_arm_R": "clavicle_R",
    "forearm_L": "upper_arm_L", "forearm_R": "upper_arm_R", "hand_L": "forearm_L", "hand_R": "forearm_R",
    "thigh_L": "pelvis", "thigh_R": "pelvis", "shin_L": "thigh_L", "shin_R": "thigh_R",
    "foot_L": "shin_L", "foot_R": "shin_R", "toe_L": "foot_L", "toe_R": "foot_R",
}  # fmt: skip
PAR = np.array([-1] + [J[PARENT[n]] for n in JOINTS[1:]])
# rest offset from the parent joint, in units of H (x forward, y left, z up)
_ARM, _LEG = (C.ARM_UPPER, C.ARM_FORE, C.ARM_HAND), (C.LEG_THIGH, C.LEG_SHIN)
OFFSET = np.zeros((21, 3))
for name, off in {
    "spine": (0, 0, 0.15), "chest": (0, 0, 0.14), "neck": (0, 0, 0.0), "head": (0, 0, 0.05),
    "clavicle_L": (0, 0.025, -0.005), "clavicle_R": (0, -0.025, -0.005),
    "upper_arm_L": (0, 0.09, 0), "upper_arm_R": (0, -0.09, 0),
    "forearm_L": (0, 0, -_ARM[0]), "forearm_R": (0, 0, -_ARM[0]),
    "hand_L": (0, 0, -_ARM[1]), "hand_R": (0, 0, -_ARM[1]),
    "thigh_L": (0, C.HIP_HALF, 0), "thigh_R": (0, -C.HIP_HALF, 0),
    "shin_L": (0, 0, -_LEG[0]), "shin_R": (0, 0, -_LEG[0]),
    "foot_L": (0, 0, -_LEG[1]), "foot_R": (0, 0, -_LEG[1]),
    "toe_L": (0.075, 0, -0.025), "toe_R": (0.075, 0, -0.025),
}.items():  # fmt: skip
    OFFSET[J[name]] = off
PELVIS_REST_Z = C.HIP_Z  # pelvis (hip centre) height standing, x H
ARM_L1, ARM_L2, HAND_L = _ARM  # upper arm, forearm, wrist-to-fingertip (x H)
LEG_L1, LEG_L2 = _LEG
ANKLE_Z = C.ANKLE_Z
PALM = 0.5 * HAND_L  # palm centre: half the wrist-to-fingertip length, along the forearm axis


# ------------------------------------------------------------------ quaternions
def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])  # fmt: skip


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    return q / np.linalg.norm(q)


def qaxis(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), *(axis * s)])


def qmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def mat_to_q(m):
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return qnorm(q)


def slerp(a, b, t):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        return qnorm(a + t * (b - a))
    th = math.acos(d)
    return (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)


def yaw_q(yaw):
    return qaxis((0, 0, 1), yaw)


IDENT = np.array([1.0, 0.0, 0.0, 0.0])


# ------------------------------------------------------------------ forward kinematics
def qmat_batch(q):
    """Rotation matrices (n,3,3) of quaternions (n,4)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def fk(H: float, pelvis_pos, local_q):
    """World joint positions (21,3) and world rotation matrices (21,3,3) from 21 local quaternions (pelvis: world)."""
    local = qmat_batch(np.asarray(local_q, float))
    off = OFFSET * H
    pos = np.empty((21, 3))
    rot = np.empty((21, 3, 3))
    pos[0] = pelvis_pos
    rot[0] = local[0]
    for j in range(1, 21):
        p = PAR[j]
        rot[j] = rot[p] @ local[j]
        pos[j] = pos[p] + rot[p] @ off[j]
    return pos, rot


def neutral_pose(yaw: float = 0.0):
    q = np.tile(IDENT, (21, 1))
    q[0] = yaw_q(yaw)
    return q
