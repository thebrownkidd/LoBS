"""Tree forward kinematics (pose + twist) for one actor.

Child body frame at q=0 is  C0 = parent o rel_pose.  A hinge rotates the child about the
joint axis through the joint point (child-frame ``pos``); a slide translates along the axis.
Twists: linear velocity is that of the body *reference origin*, angular is world omega.
"""

from __future__ import annotations

import numpy as np

from . import math3d as m3


def fk_actor(reg, actor_idx: int, actor_pose, q, qd, root_v, root_w):
    """Return {global_body: (pos, quat, linvel, angvel)} for all bodies of one actor.

    ``actor_pose`` is a Pose/(pos, quat); ``q``/``qd`` are indexed by joint DOF (registry order).
    """
    if hasattr(actor_pose, "pos"):
        a_pos, a_quat = np.array(actor_pose.pos), np.array(actor_pose.quat)
    else:
        a_pos, a_quat = np.asarray(actor_pose[0], float), np.asarray(actor_pose[1], float)
    out = {}
    for gi in reg.actors[actor_idx].bodies:
        b = reg.bodies[gi]
        if b.parent < 0:
            p, qt = m3.pose_compose(a_pos, a_quat, b.rel_pos, b.rel_quat)
            out[gi] = (p, qt, np.asarray(root_v, float), np.asarray(root_w, float))
            continue
        pp, pq, pv, pw = out[b.parent]
        c0_p, c0_q = m3.pose_compose(pp, pq, b.rel_pos, b.rel_quat)
        c0_v = pv + np.cross(pw, c0_p - pp)
        j = reg.joints[b.joint]
        if j.type == "fixed":
            out[gi] = (c0_p, c0_q, c0_v, pw)
            continue
        if j.type == "ball":
            # q = rotation vector of the child relative to its zero pose, applied in the child frame;
            # qd = relative angular velocity expressed in the child frame (MuJoCo's ball-joint convention)
            quat = m3.quat_mul(c0_q, m3.quat_exp(q[j.dof : j.dof + 3]))
            rc = m3.quat_to_mat(quat)
            jp = c0_p + m3.quat_rotate(c0_q, j.pos)
            p = jp - rc @ j.pos
            w = pw + rc @ qd[j.dof : j.dof + 3]
            v = pv + np.cross(pw, jp - pp) + np.cross(w, p - jp)
            out[gi] = (p, quat, v, w)
            continue
        axis_w = m3.quat_rotate(c0_q, j.axis)
        qv, qdv = q[j.dof], qd[j.dof]
        if j.type == "hinge":
            jp = c0_p + m3.quat_rotate(c0_q, j.pos)
            rq = m3.quat_from_axis_angle(axis_w, qv)
            p = jp + m3.quat_rotate(rq, c0_p - jp)
            quat = m3.quat_mul(rq, c0_q)
            vj = pv + np.cross(pw, jp - pp)
            v = vj + np.cross(pw, p - jp) + np.cross(axis_w * qdv, p - jp)
            w = pw + axis_w * qdv
            out[gi] = (p, quat, v, w)
        elif j.type == "slide":
            p = c0_p + axis_w * qv
            v = c0_v + np.cross(pw, axis_w * qv) + axis_w * qdv
            out[gi] = (p, c0_q, v, pw)
        else:
            raise NotImplementedError(f"joint type '{j.type}'")
    return out
