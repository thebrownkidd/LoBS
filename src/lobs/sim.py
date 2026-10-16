"""Simulation: a live run bound to one backend."""

from __future__ import annotations

import dataclasses
import types

import numpy as np

from . import backends as backend_registry
from .backends.base import PhysicsBackend
from .backends.capabilities import UNSUPPORTED, Capability
from .errors import LobsError, WorldValidationError
from .measure import build_evaluator
from .measure.context import EvalCtx
from .registry import build_registry, resolve_initial
from .state import StateFrame


class History:
    """Recorded measure values: ``history[name]`` -> array (T, *shape); ``history.time`` -> (T,)."""

    def __init__(self, names, limit):
        self._limit = limit
        self._t: list = []
        self._v: dict[str, list] = {n: [] for n in names}

    def _append(self, t, values):
        self._t.append(t)
        for n, lst in self._v.items():
            lst.append(values[n])
        if self._limit and len(self._t) > self._limit:
            del self._t[0]
            for lst in self._v.values():
                del lst[0]

    def clear(self):
        self._t.clear()
        for lst in self._v.values():
            lst.clear()

    @property
    def time(self) -> np.ndarray:
        return np.asarray(self._t)

    def __getitem__(self, name) -> np.ndarray:
        if name not in self._v:
            raise KeyError(f"no measure '{name}'; known: {sorted(self._v)}")
        return np.asarray(self._v[name])

    def __contains__(self, name):
        return name in self._v

    def __len__(self):
        return len(self._t)

    def names(self):
        return list(self._v)


class Simulation:
    def __init__(self, spec, reg, backend: PhysicsBackend, evaluator, *, seed=None, record=True,
                 record_every=1, record_contacts=True, history_limit=None, warnings=()):  # fmt: skip
        self.spec, self.reg, self.backend, self.evaluator = spec, reg, backend, evaluator
        self.record, self.record_every = record, max(1, int(record_every))
        self.record_contacts = record_contacts
        self.history = History(evaluator.outputs, history_limit)
        self._warnings = list(warnings)
        self.tick_dt = spec.timing.dt * spec.timing.substeps
        self._caps = type(backend).capabilities()
        self._contacts_ok = self._caps.get(Capability.CONTACTS, UNSUPPORTED).ok
        self._need_contacts = "contacts" in evaluator.needs
        self._need_accel = "accel" in evaluator.needs
        self._frames: list[StateFrame] = []
        self._actions: list[np.ndarray] = []
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tick = 0
        self.closed = False
        self.reset(seed)

    # ------------------------------------------------------------ construction
    @classmethod
    def create(cls, world, backend="mujoco", *, seed=None, record=True, record_every=1,
               record_contacts=True, history_limit=None, options=None) -> Simulation:  # fmt: skip
        spec = world.spec.model_copy(deep=True)
        probe = type(world)(spec=spec, base_dir=world.base_dir)
        if isinstance(backend, str):
            factory = backend_registry.get(backend)
            opts = options if options is not None else getattr(spec.backend_options, backend, None)
            report = probe.validate(backend)
        else:
            factory = backend
            opts = options
            report = probe.validate(backend if isinstance(backend, type) else None)
        if not report.ok:
            raise WorldValidationError(report)
        reg = build_registry(spec, world.base_dir)
        evaluator = build_evaluator(reg, spec.measures)
        be = factory(spec, reg, opts)
        return cls(spec, reg, be, evaluator, seed=seed, record=record, record_every=record_every,
                   record_contacts=record_contacts, history_limit=history_limit,
                   warnings=[i.message for i in report.warnings])  # fmt: skip

    # ------------------------------------------------------------ properties
    @property
    def time(self) -> float:
        return self.tick * self.tick_dt

    @property
    def done(self) -> bool:
        d = self.spec.timing.duration
        return d is not None and self.time >= d - 1e-12

    @property
    def action_names(self) -> list[str]:
        return [a.path for a in self.reg.actuators]

    @property
    def capabilities(self):
        return self._caps

    @property
    def info(self) -> dict:
        info = dict(self.backend.info())
        info["warnings"] = list(self._warnings)
        info["tick_dt"] = self.tick_dt
        return info

    @property
    def native(self):
        return self.backend.native

    # ------------------------------------------------------------ core loop
    def reset(self, seed=None, initial_state=None):
        if seed is not None:
            self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.backend.reset(resolve_initial(self.reg, initial_state), self.seed)
        self.evaluator.reset()
        self.history.clear()
        self._frames.clear()
        self._actions.clear()
        self._wrenches: list = []
        self._action = np.zeros(len(self.reg.actuators))
        self.tick = 0
        self._observe(initial=True)
        return self.measurements()

    def _read(self) -> StateFrame:
        want_c = self._need_contacts or (self.record and self.record_contacts and self._contacts_ok)
        f = self.backend.read_state(contacts=want_c, accelerations=self._need_accel)
        return dataclasses.replace(f, t=self.tick * self.tick_dt)

    def _observe(self, initial=False):
        frame = self._read()
        self._last_frame = frame
        self._values = self.evaluator.evaluate(frame)
        if self.record and (self.tick % self.record_every == 0):
            self.history._append(frame.t, self._values)
            self._frames.append(frame)
            self._actions.append(self._action.copy())

    def _map_action(self, action) -> np.ndarray:
        n = len(self.reg.actuators)
        if isinstance(action, dict):
            u = self._action.copy()
            for k, v in action.items():
                u[self.reg.resolve_actuator(k)] = v
            return u
        u = np.asarray(action, dtype=float).reshape(-1)
        if u.shape != (n,):
            raise LobsError(
                f"action must have {n} entries (order: {self.action_names}), got {u.shape[0]}"
            )
        return u

    def step(self, action=None, n: int = 1):
        if self.closed:
            raise LobsError("simulation is closed")
        for _ in range(n):
            if action is not None:
                self._action = self._map_action(action)
                self.backend.set_actuator_targets(self._action)
            self._apply_wrenches()
            self.backend.step(self.spec.timing.substeps)
            self.tick += 1
            self._observe()
        return self.measurements()

    def run(self, seconds=None, ticks=None, controller=None):
        if ticks is None:
            seconds = seconds if seconds is not None else self.spec.timing.duration
            if seconds is None:
                raise LobsError("run() needs seconds=/ticks= or timing.duration")
            ticks = int(round(seconds / self.tick_dt))
        for _ in range(ticks):
            self.step(controller(self) if controller is not None else None)
        return self.measurements()

    # ------------------------------------------------------------ forces
    def apply_force(self, thing, force, torque=None, at=None, ticks: int = 1):
        """Disturbance in world frame on an actor's root body or on a body; held for ``ticks`` ticks."""
        th = self.reg.resolve(thing)
        if th.kind == "actor":
            body = self.reg.actors[th.index].bodies[0]
        elif th.kind == "body":
            body = th.index
        else:
            raise LobsError(f"cannot apply a force to a {th.kind}")
        if self.reg.bodies[body].static:
            raise LobsError(f"'{self.reg.bodies[body].path}' is static")
        self._wrenches.append([body, np.asarray(force, float), np.zeros(3) if torque is None else np.asarray(torque, float),
                               None if at is None else np.asarray(at, float), int(ticks)])  # fmt: skip

    def _apply_wrenches(self):
        keep = []
        ctx = EvalCtx(self.reg, self._last_frame)
        for w in self._wrenches:
            body, f, tau, at, left = w
            point = ctx.body_com(body)[0] if at is None else at
            self.backend.add_wrench(body, f, tau, point)
            if left > 1:
                w[4] = left - 1
                keep.append(w)
        self._wrenches = keep

    # ------------------------------------------------------------ queries
    def measurements(self, names=None) -> types.MappingProxyType:
        vals = self._values if names is None else {n: self._values[n] for n in names}
        return types.MappingProxyType(dict(vals))

    def __getitem__(self, name):
        return self._values[name]

    def state(self) -> StateFrame:
        return self._last_frame

    # ------------------------------------------------------------ export
    def trajectory(self, measures=None):
        from .trajectory import Trajectory

        return Trajectory.from_sim(self, measures)

    def export(self, path, measures=None):
        return self.trajectory(measures).save(path)

    def close(self):
        if not self.closed:
            self.backend.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
