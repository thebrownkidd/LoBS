"""Engine-independent per-tick state: the primitives every measure is computed from."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from . import math3d as m3


@dataclass(frozen=True)
class ContactSet:
    """Flat contact arrays. Body indices are registry-global; ``body_a < body_b``.

    ``normal`` is a world unit vector from A to B; ``depth >= 0`` is penetration;
    ``force_on_b`` is the world-frame force on B; ``normal_force >= 0``.
    """

    body_a: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    body_b: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    point: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    normal: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    depth: np.ndarray = field(default_factory=lambda: np.zeros(0))
    force_on_b: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    normal_force: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return len(self.body_a)


@dataclass(frozen=True)
class StateFrame:
    """One tick. Arrays cover the D *dynamic* bodies (``registry.dynamic_order``) and J DOF joints."""

    t: float
    body_pos: np.ndarray  # (D, 3) reference-frame origin, world
    body_quat: np.ndarray  # (D, 4)
    body_linvel: np.ndarray  # (D, 3) velocity of the reference origin
    body_angvel: np.ndarray  # (D, 3) world
    joint_q: np.ndarray  # (J,)
    joint_qd: np.ndarray
    joint_effort: np.ndarray
    body_linacc: np.ndarray | None = None  # coordinate acceleration of the reference origin
    body_angacc: np.ndarray | None = None
    joint_qdd: np.ndarray | None = None
    contacts: ContactSet | None = None
    ball_slots: tuple = ()  # first DOF index of each ball joint (3 slots: rotation vector / child-frame omega)

    def advanced(self, delta: float) -> StateFrame:
        """Kinematic extrapolation along the twists (used for exact rates). Drops contacts/accels."""
        quat = m3.quat_mul(m3.quat_exp(delta * self.body_angvel), self.body_quat)
        jq = self.joint_q + delta * self.joint_qd
        for s in (
            self.ball_slots
        ):  # rotation vectors compose on the right: q <- log(exp(q) * exp(delta*omega))
            jq[s : s + 3] = m3.quat_log(
                m3.quat_mul(
                    m3.quat_exp(self.joint_q[s : s + 3]),
                    m3.quat_exp(delta * self.joint_qd[s : s + 3]),
                )
            )
        return replace(
            self,
            t=self.t + delta,
            body_pos=self.body_pos + delta * self.body_linvel,
            body_quat=m3.quat_normalize(quat),
            joint_q=jq,
            body_linacc=None,
            body_angacc=None,
            joint_qdd=None,
            contacts=None,
        )
