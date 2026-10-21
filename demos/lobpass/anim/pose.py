"""Pose solver: targets in, 21 local quaternions out.

Legs and arms are solved with two-bone IK so planted feet and ball-holding hands land exactly where the
animation layers ask. If a planted foot is out of reach the pelvis is lowered (a foot never floats)."""

from __future__ import annotations

import math

import numpy as np

from . import ik, rig

LEG_REACH = (rig.LEG_L1 + rig.LEG_L2) * 0.998
ARM_REACH = (rig.ARM_L1 + rig.ARM_L2) * 0.998


def euler_q(yaw=0.0, pitch=0.0, roll=0.0):
    """Z (yaw) * Y (pitch, positive = lean forward/toes down) * X (roll)."""
    q = rig.qmul(rig.yaw_q(yaw), rig.qaxis((0, 1, 0), pitch))
    return rig.qmul(q, rig.qaxis((1, 0, 0), roll))


def solve_pose(H: float, p: dict):
    """p keys: pelvis_pos (3), pelvis_q (world), spine_q, chest_q (local, optional), feet[2] = (ankle_world (3),
    yaw, pitch), arms[2] = palm world target (3) or None (hang), head = (yaw, pitch) relative to the chest,
    clav = (left, right) elevation (rad, optional). Returns (local_q (21,4), pelvis_pos (3), info)."""
    q = np.tile(rig.IDENT, (21, 1))
    q[rig.J["pelvis"]] = p["pelvis_q"]
    q[rig.J["spine"]] = p.get("spine_q", rig.IDENT)
    q[rig.J["chest"]] = p.get("chest_q", rig.IDENT)
    pelvis = np.array(p["pelvis_pos"], float)
    # --- legs: lower the pelvis until both ankles are reachable
    for _ in range(4):
        pos, rot = rig.fk(H, pelvis, q)
        worst = 0.0
        for i, side in enumerate("LR"):
            hip = pos[rig.J[f"thigh_{side}"]]
            worst = max(worst, float(np.linalg.norm(p["feet"][i][0] - hip)) - LEG_REACH * H)
        if worst <= 1e-6:
            break
        if pelvis[2] < p["pelvis_pos"][2] - 0.30 * H:  # an impossible foot target: let the IK clamp instead
            break
        pelvis = pelvis - np.array([0.0, 0.0, worst * 1.02 + 1e-4])
    pos, rot = rig.fk(H, pelvis, q)
    fwd = rot[0][:, 0]
    info = {"knee_reached": [True, True]}
    for i, side in enumerate("LR"):
        ankle, fyaw, fpitch = p["feet"][i]
        hip = pos[rig.J[f"thigh_{side}"]]
        pole = ik.unit(fwd + np.array([0.0, 0.25 if side == "L" else -0.25, 0.0]))
        mid, end, ok = ik.solve_mid(hip, ankle, rig.LEG_L1 * H, rig.LEG_L2 * H, pole)
        info["knee_reached"][i] = ok
        qt, qs, r1 = ik.limb_quats(rot[0], hip, mid, end, pole)
        q[rig.J[f"thigh_{side}"]], q[rig.J[f"shin_{side}"]] = qt, qs
        r_shin = r1 @ rig.qmat(qs)
        r_foot = rig.qmat(rig.qmul(rig.yaw_q(fyaw), rig.qaxis((0, 1, 0), fpitch)))
        q[rig.J[f"foot_{side}"]] = rig.mat_to_q(r_shin.T @ r_foot)
    # --- head and neck share the look rotation
    hy, hp = p.get("head", (0.0, 0.0))
    hy, hp = float(np.clip(hy, -math.radians(70), math.radians(70))), float(np.clip(hp, -math.radians(40), math.radians(40)))
    q[rig.J["neck"]] = euler_q(hy * 0.4, hp * 0.4)
    q[rig.J["head"]] = euler_q(hy * 0.6, hp * 0.6)
    # --- arms: raise the clavicle toward high targets, then IK the wrist so the palm centre hits the target
    pos, rot = rig.fk(H, pelvis, q)
    arms = p.get("arms", (None, None))
    clav = list(p.get("clav", (0.0, 0.0)))
    for i, side in enumerate("LR"):
        tgt = arms[i]
        if tgt is not None and "clav" not in p:
            shoulder = pos[rig.J[f"upper_arm_{side}"]]
            clav[i] = math.radians(70.0) * ik.smoothstep((tgt[2] - shoulder[2]) / (0.45 * H))
    for i, side in enumerate("LR"):
        q[rig.J[f"clavicle_{side}"]] = rig.qaxis((1, 0, 0), clav[i] if side == "L" else -clav[i])
    pos, rot = rig.fk(H, pelvis, q)
    info["palm"] = [None, None]
    for i, side in enumerate("LR"):
        shoulder = pos[rig.J[f"upper_arm_{side}"]]
        r_clav = rot[rig.J[f"clavicle_{side}"]]
        tgt = arms[i]
        if tgt is None:  # hanging, slightly bent
            tgt = shoulder + rot[rig.J["chest"]] @ np.array([0.02, 0.02 if side == "L" else -0.02, -0.34]) * H
        tgt = np.asarray(tgt, float)
        chest = rot[rig.J["chest"]]
        pole = ik.unit(-chest[:, 0] + chest[:, 2] * -0.3 + chest[:, 1] * (0.5 if side == "L" else -0.5))
        wrist_t = tgt
        for _ in range(3):  # the palm centre sits PALM along the forearm beyond the wrist
            elbow, _, _ = ik.solve_mid(shoulder, wrist_t, rig.ARM_L1 * H, rig.ARM_L2 * H, pole)
            d2 = ik.unit(tgt - elbow)
            wrist_t = tgt - d2 * rig.PALM * H
        elbow, wrist, ok = ik.solve_mid(shoulder, wrist_t, rig.ARM_L1 * H, rig.ARM_L2 * H, pole)
        qu, ql, r1 = ik.limb_quats(r_clav, shoulder, elbow, wrist, pole)
        q[rig.J[f"upper_arm_{side}"]], q[rig.J[f"forearm_{side}"]] = qu, ql
        d_fore = ik.unit(wrist - elbow)
        info["palm"][i] = wrist + d_fore * rig.PALM * H
    return q, pelvis, info
