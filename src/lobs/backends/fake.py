"""FakeBackend: replays analytic motion so measures can be tested without a physics engine."""

from __future__ import annotations

import numpy as np

from ..state import ContactSet, StateFrame
from .base import PhysicsBackend
from .capabilities import NATIVE, Capability


class FakeBackend(PhysicsBackend):
    name = "fake"

    @classmethod
    def available(cls) -> bool:
        return True

    @classmethod
    def capabilities(cls):
        return {c: NATIVE for c in Capability}

    @classmethod
    def engine_version(cls) -> str:
        return "fake-1"

    def __init__(self, spec, registry, options=None, motion=None, contacts=None):
        """``motion(t) -> dict`` with arrays pos (D,3), quat (D,4), linvel, angvel and optionally
        q, qd, effort, linacc, angacc, qdd. ``contacts(t) -> ContactSet`` is optional."""
        self.reg, self.dt = registry, spec.timing.dt
        self.motion, self.contacts_fn = motion, contacts
        self.t = 0.0
        self.targets = np.zeros(len(registry.actuators))
        self.wrenches: list = []

    def reset(self, initial, seed):
        self.t = 0.0
        self.targets = np.zeros(len(self.reg.actuators))
        self.wrenches = []

    def set_actuator_targets(self, u):
        self.targets = np.asarray(u, dtype=float).copy()

    def add_wrench(self, body, force_w, torque_w, point_w):
        self.wrenches.append((body, np.asarray(force_w), np.asarray(torque_w), np.asarray(point_w)))

    def step(self, substeps):
        self.t += substeps * self.dt
        self.wrenches = []

    def read_state(self, *, contacts, accelerations):
        m = self.motion(self.t)
        nd, nj = len(self.reg.dynamic_order), self.reg.n_dof
        z3, zj = np.zeros((nd, 3)), np.zeros(nj)
        return StateFrame(
            t=self.t,
            body_pos=np.asarray(m["pos"], float).reshape(nd, 3),
            body_quat=np.asarray(m["quat"], float).reshape(nd, 4),
            body_linvel=np.asarray(m["linvel"], float).reshape(nd, 3),
            body_angvel=np.asarray(m["angvel"], float).reshape(nd, 3),
            joint_q=np.asarray(m.get("q", zj), float),
            joint_qd=np.asarray(m.get("qd", zj), float),
            joint_effort=np.asarray(m.get("effort", zj), float),
            body_linacc=np.asarray(m["linacc"], float).reshape(nd, 3) if accelerations and "linacc" in m else (z3 if accelerations else None),
            body_angacc=np.asarray(m["angacc"], float).reshape(nd, 3) if accelerations and "angacc" in m else (z3 if accelerations else None),
            joint_qdd=np.asarray(m.get("qdd", zj), float) if accelerations else None,
            contacts=(self.contacts_fn(self.t) if self.contacts_fn else ContactSet()) if contacts else None,
            ball_slots=tuple(self.reg.joints[i].dof for i in self.reg.dof_joints if self.reg.joints[i].type == "ball"),
        )  # fmt: skip

    def info(self):
        return {"engine": "fake", "version": self.engine_version(), "approximations": []}

    @property
    def native(self):
        return None
