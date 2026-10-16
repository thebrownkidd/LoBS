"""EvalCtx: resolves poses/twists of worlds, bodies, actors and anchors from one StateFrame."""

from __future__ import annotations

import numpy as np

from .. import math3d as m3
from ..anchors import rule_offset
from ..errors import MeasureError
from ..registry import Registry, Thing
from ..state import StateFrame

PERTURB = 1e-6  # seconds, for exact rates / anchor velocities (see plan 8.5)
_I3 = np.eye(3)


class EvalCtx:
    def __init__(self, reg: Registry, frame: StateFrame, functions: dict | None = None):
        self.reg, self.frame, self.functions = reg, frame, functions or {}
        self.t = frame.t
        self._body: dict = {}
        self._twist: dict = {}
        self._anchor: dict = {}
        self._avel: dict = {}

    def advanced(self, delta: float) -> EvalCtx:
        return EvalCtx(self.reg, self.frame.advanced(delta), self.functions)

    # ------------------------------------------------------------ bodies
    def body_pose(self, b: int):
        """(p, q, R) of body reference frame in world."""
        hit = self._body.get(b)
        if hit is None:
            info = self.reg.bodies[b]
            if info.static:
                p, q = info.world_pos, info.world_quat
            else:
                p, q = self.frame.body_pos[info.dyn], self.frame.body_quat[info.dyn]
            hit = (p, q, m3.quat_to_mat(q))
            self._body[b] = hit
        return hit

    def body_twist(self, b: int):
        """(v of reference origin, omega) in world."""
        info = self.reg.bodies[b]
        if info.static:
            return np.zeros(3), np.zeros(3)
        return self.frame.body_linvel[info.dyn], self.frame.body_angvel[info.dyn]

    def body_com(self, b: int):
        """(p_com, v_com) in world."""
        p, _, r = self.body_pose(b)
        v, w = self.body_twist(b)
        rc = r @ self.reg.bodies[b].com
        return p + rc, v + m3.cross3(w, rc)

    # ------------------------------------------------------------ things
    def pose(self, thing: Thing):
        """(p, R, q) of a thing's frame in world."""
        k = thing.kind
        if k == "world":
            return np.zeros(3), _I3, m3.IDENTITY_QUAT
        if k == "body":
            p, q, r = self.body_pose(thing.index)
            if thing.point == "com":
                p = p + r @ self.reg.bodies[thing.index].com
            return p, r, q
        if k == "actor":
            actor = self.reg.actors[thing.index]
            root = actor.bodies[0]
            p, q, r = self.body_pose(root)
            if thing.point == "com" and actor.mass > 0:
                p = (
                    sum(self.reg.bodies[b].mass * self.body_com(b)[0] for b in actor.bodies)
                    / actor.mass
                )
            return p, r, q
        if k == "anchor":
            return self._anchor_pose(thing.index)
        raise MeasureError(f"'{thing.path}' ({k}) has no pose")

    def twist(self, thing: Thing):
        """(velocity of the thing's point, angular velocity) in world."""
        k = thing.kind
        if k == "world":
            return np.zeros(3), np.zeros(3)
        if k == "body":
            v, w = self.body_twist(thing.index)
            if thing.point == "com":
                v = self.body_com(thing.index)[1]
            return v, w
        if k == "actor":
            actor = self.reg.actors[thing.index]
            root = actor.bodies[0]
            v, w = self.body_twist(root)
            if thing.point == "com" and actor.mass > 0:
                v = (
                    sum(self.reg.bodies[b].mass * self.body_com(b)[1] for b in actor.bodies)
                    / actor.mass
                )
            return v, w
        if k == "anchor":
            return self._anchor_twist(thing.index)
        raise MeasureError(f"'{thing.path}' ({k}) has no twist")

    # ------------------------------------------------------------ anchors
    def _anchor_pose(self, ai: int):
        hit = self._anchor.get(ai)
        if hit is not None:
            return hit
        a = self.reg.anchors[ai]
        if a.parent < 0:
            pb, qb, rb = np.zeros(3), m3.IDENTITY_QUAT, _I3
        else:
            pb, qb, rb = self.body_pose(a.parent)
        local = a.pos + rule_offset(a.rule, self.t)
        p = pb + rb @ local
        if getattr(a.rule, "type", "rigid") == "translate_only":
            q = a.quat
        else:
            q = m3.quat_mul(qb, a.quat)
        hit = (p, m3.quat_to_mat(q), q)
        self._anchor[ai] = hit
        return hit

    def _anchor_twist(self, ai: int):
        hit = self._avel.get(ai)
        if hit is None:
            plus, minus = self.advanced(PERTURB), self.advanced(-PERTURB)
            pp, _, qp = plus._anchor_pose(ai)
            pm, _, qm = minus._anchor_pose(ai)
            v = (pp - pm) / (2 * PERTURB)
            w = m3.quat_log(m3.quat_mul(qp, m3.quat_conj(qm))) / (2 * PERTURB)
            hit = (v, w)
            self._avel[ai] = hit
        return hit
