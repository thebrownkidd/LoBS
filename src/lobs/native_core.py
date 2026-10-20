"""Native tick core: runs force laws, state-machine behaviors, the physics step and the measure program in
one native call per tick (src/native). It is an optimisation of ``Simulation.step`` with identical semantics;
simulations it cannot take (script behaviors, contacts / accelerations needed, non-MuJoCo backends, ball
joints, extension not built) stay on the Python path."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import native
from .errors import LobsError
from .measure.native_program import THING_KIND
from .schema import behavior as S

MODES = {"seek": 0, "arrive": 1, "flee": 2, "intercept": 3, "hold": 4}
HIDDEN_BEH = "__beh."
HIDDEN_FORCE = "__force."


def thing_tuple(th):
    return (THING_KIND[th.kind], int(th.index), th.point == "com")


NO_THING = (0, 0, True)


@dataclass
class RuntimeInfo:
    """What the exporter needs to map event / behavior / force measures onto the native runtime."""

    events: dict
    behaviors: list
    forces: list


def collect_events(spec, evaluator) -> dict:
    names = []
    for b in spec.behaviors:
        for _, a in b.actions():
            if isinstance(a, S.Emit) and a.event not in names:
                names.append(a.event)
    for n in evaluator.nodes:
        sp = getattr(n, "spec", None)
        if sp is not None and getattr(sp, "op", None) == "event" and str(sp.args["name"]) not in names:
            names.append(str(sp.args["name"]))
    return {n: i for i, n in enumerate(names)}


class LawView:
    """The state of one native force law, shaped like ``forces.Law``."""

    def __init__(self, core, i, law):
        self._core, self._i = core, i
        self.name, self.on, self.toward = law.name, law.on, law.toward
        self.adversary = law.adversary

    def _v(self):
        return self._core.rt.law_view(self._i)

    impulse = property(lambda s: s._v()[0])
    active_time = property(lambda s: s._v()[1])
    peak = property(lambda s: s._v()[2])
    active = property(lambda s: bool(s._v()[3]))
    last_force = property(lambda s: np.array(s._v()[4]))

    @property
    def enabled(self):
        return bool(self._v()[5])

    @enabled.setter
    def enabled(self, flag):
        self._core.rt.set_law_enabled(self._i, bool(flag))

    def reset(self):  # the runtime resets all laws together
        pass

    def stats(self) -> dict:
        v = self._v()
        return {"impulse": v[0], "active_time": v[1], "peak": v[2], "last_force": np.array(v[4]), "active": bool(v[3])}


class BehaviorView:
    """The state of one native state-machine behavior, shaped like ``behavior._B``."""

    def __init__(self, core, i, b):
        self._core, self._i = core, i
        self.spec, self.machine, self.on, self.owned = b.spec, b.machine, b.on, b.owned
        self.state_names = b.state_names
        self.script_obj = None
        m = b.machine
        self._var_names = list(m.variables)
        self._var_scalar = [isinstance(v, (int, float)) for v in m.variables.values()]

    def _v(self):
        return self._core.rt.behavior_view(self._i)

    state_idx = property(lambda s: s._v()[0])
    state_name = property(lambda s: s.state_names[s._v()[0]])
    enter_t = property(lambda s: s._v()[1])

    @property
    def enabled(self):
        return bool(self._v()[2])

    @enabled.setter
    def enabled(self, flag):
        self._core.rt.set_behavior_enabled(self._i, bool(flag))

    @property
    def vars(self) -> dict:
        flat, out, off = self._v()[3], {}, 0
        for name, scalar in zip(self._var_names, self._var_scalar, strict=True):
            n = 1 if scalar else 3
            out[name] = float(flat[off]) if scalar else np.array(flat[off : off + 3])
            off += n
        return out


class _EventView:
    def __init__(self, core):
        self._core = core

    def __contains__(self, name):
        i = self._core.events.get(name)
        return i is not None and i in self._core.rt.events()

    def __iter__(self):
        names = {i: n for n, i in self._core.events.items()}
        return iter([names[i] for i in self._core.rt.events()])

    def __len__(self):
        return len(self._core.rt.events())


class NativeBehaviorRuntime:
    """Drop-in for ``behavior.BehaviorRuntime`` (state lives in the native runtime)."""

    def __init__(self, core, py_runtime):
        self.core, self.reg, self.tick_dt = core, py_runtime.reg, py_runtime.tick_dt
        self.items = [BehaviorView(core, i, b) for i, b in enumerate(py_runtime.items)]
        self.by_name = {b.spec.name: b for b in self.items}
        self.names = self.by_name
        self.owned = py_runtime.owned
        self.laws = {v.name: v for v in core.law_views}
        self.events = _EventView(core)
        self._log_n, self._log = 0, []

    @property
    def behaviors(self):
        return self.names

    @property
    def forces(self):
        return self.laws

    @property
    def log(self) -> list:
        raw = self.core.rt.get_log()
        if len(raw) < self._log_n:
            self._log_n, self._log = 0, []
        for t, bi, fr, to in raw[self._log_n :]:
            b = self.items[bi]
            self._log.append((t, b.spec.name, b.state_names[fr], b.state_names[to]))
        self._log_n = len(raw)
        return self._log

    def states(self) -> dict:
        return {b.spec.name: b.state_idx for b in self.items}


class NativeCore:
    def __init__(self, sim, rt, events, law_views):
        self.sim, self.rt, self.events, self.law_views = sim, rt, events, law_views
        self.kernel = sim.backend._kernel
        self.owned = list(sim._beh.owned) if sim._beh is not None else []

    # ------------------------------------------------------------------ construction
    @staticmethod
    def build(sim):
        from .behavior import HIDDEN as BH
        from .forces import HIDDEN as FH
        from .measure import build_evaluator

        nat = native.module()
        be, spec, reg, ev = sim.backend, sim.spec, sim.reg, sim.evaluator
        if nat is None or sim.record_contacts or not hasattr(be, "_prep_read"):
            return None
        if not getattr(be, "_prepared", False):
            be._prep_read()
        if getattr(be, "_kernel", None) is None or ev._prog is None:
            return None
        if "contacts" in ev.needs or "accel" in ev.needs:
            return None
        if any(b.script is not None for b in spec.behaviors) or sim._named is None:
            return None
        events = collect_events(spec, ev)
        info = RuntimeInfo(events, list(spec.behaviors), list(spec.forces))
        try:
            ev2 = build_evaluator(reg, sim._named, native_runtime=info)
        except Exception:  # noqa: BLE001
            return None
        if ev2._prog is None:
            return None
        hidden = [n for n in ev2.outputs if n.startswith((FH, BH))]
        if any(ev2.outputs[n] not in ev2._native for n in hidden):
            return None
        sim.evaluator = ev2
        rt = nat.Runtime()
        rt.configure(ev2._prog, be._kernel, float(sim.tick_dt), int(spec.timing.substeps), [int(x) for x in be._mj_body], len(events), LobsError)  # fmt: skip
        out = ev2.outputs
        law_views = []
        for i, law in enumerate(sim._laws):
            f = spec.forces[i]
            key = f"{FH}{law.name}"
            axis_thing, axis_local = (law.axis if law.axis is not None else (None, None))
            rt.add_law(
                law.name, thing_tuple(law.on), law.toward is not None, thing_tuple(law.toward) if law.toward is not None else NO_THING,
                law.direction is not None, [float(x) for x in (law.direction if law.direction is not None else (0, 0, 0))],
                law.axis is not None, thing_tuple(axis_thing) if axis_thing is not None else NO_THING,
                [float(x) for x in (axis_local if axis_local is not None else (0, 0, 0))],
                bool(law.reaction), bool(law.has_vector), bool(law.has_when),
                out[key] if not law.has_vector else -1, out[f"{key}.vec"] if law.has_vector else -1,
                out[f"{key}.when"] if law.has_when else -1,
                law.peak_limit is not None, float(law.peak_limit or 0.0), law.impulse_limit is not None, float(law.impulse_limit or 0.0),
                law.active_time_limit is not None, float(law.active_time_limit or 0.0),
            )  # fmt: skip
            law_views.append((i, law))
            del f
        py_beh = sim._beh
        force_index = {law.name: i for i, law in enumerate(sim._laws)}
        beh_index = {b.spec.name: i for i, b in enumerate(py_beh.items)} if py_beh is not None else {}
        if py_beh is not None:
            for bi, b in enumerate(py_beh.items):
                m = b.machine
                names = b.state_names
                var_n = [1 if isinstance(v, (int, float)) else 3 for v in m.variables.values()]
                flat = []
                for v in m.variables.values():
                    flat += [float(v)] if isinstance(v, (int, float)) else [float(x) for x in v]
                var_index = {k: i for i, k in enumerate(m.variables)}
                rt.add_behavior(b.spec.name, b.on is not None, thing_tuple(b.on) if b.on is not None else NO_THING,
                                int(py_beh.period[bi]), int(py_beh.latency[bi]), names.index(m.initial or names[0]),
                                bool(b.spec.enabled), var_n, flat, len(names))  # fmt: skip
                for key, a in b.spec.actions():
                    sname, phase_name, _ = key.split(".")[0], key.split(".")[1], key
                    si, pi = names.index(sname), {"on_enter": 0, "do": 1, "on_exit": 2}[phase_name]
                    kw = dict(act_idx=-1, node=-1, has_on=False, on=NO_THING, has_max=False, max_force=0.0, mode=0, has_target=False,
                              target=NO_THING, speed=1.0, max_accel=5.0, gain=4.0, stop_radius=0.0, plane=True, var=-1, event=-1, idx=-1, flag=True)  # fmt: skip
                    hname = f"{BH}{b.spec.name}.{key}"
                    if isinstance(a, S.Actuate):
                        typ = 0
                        kw.update(act_idx=reg.resolve_actuator(a.actuator), node=out[f"{hname}.value"])
                    elif isinstance(a, S.Force):
                        typ = 1
                        kw.update(node=out[f"{hname}.vector"], has_on=a.on is not None, on=thing_tuple(reg.resolve(a.on)) if a.on else NO_THING,
                                  has_max=a.max_force is not None, max_force=float(a.max_force or 0.0))  # fmt: skip
                    elif isinstance(a, S.Steer):
                        typ = 2
                        kw.update(mode=MODES[a.mode], has_on=a.on is not None, on=thing_tuple(reg.resolve(a.on)) if a.on else NO_THING,
                                  has_target=a.target is not None, target=thing_tuple(reg.resolve(a.target)) if a.target else NO_THING,
                                  speed=float(a.speed), max_accel=float(a.max_accel), gain=float(a.gain), plane=bool(a.plane),
                                  stop_radius=float(a.stop_radius))  # fmt: skip
                    elif isinstance(a, S.SetVar):
                        typ = 3
                        kw.update(var=var_index[a.name], node=out[f"{hname}.value"])
                    elif isinstance(a, S.Emit):
                        typ = 4
                        kw.update(event=events[a.event])
                    elif isinstance(a, S.EnableForce):
                        typ = 5
                        kw.update(idx=force_index[a.force], flag=bool(a.enabled))
                    elif isinstance(a, S.EnableBehavior):
                        typ = 6
                        kw.update(idx=beh_index[a.behavior], flag=bool(a.enabled))
                    else:
                        return None
                    rt.add_action(bi, si, pi, typ, kw["act_idx"], kw["node"], kw["has_on"], kw["on"], kw["has_max"], kw["max_force"],
                                  kw["mode"], kw["has_target"], kw["target"], kw["speed"], kw["max_accel"], kw["gain"],
                                  kw["stop_radius"], kw["plane"], kw["var"], kw["event"], kw["idx"], kw["flag"])  # fmt: skip
                for ti, tr in enumerate(m.transitions):
                    rt.add_transition(bi, names.index(tr.frm), names.index(tr.to), out[f"{BH}{b.spec.name}.t{ti}.when"],
                                      float(tr.min_time), int(tr.priority))  # fmt: skip
        rt.finalize()
        core = NativeCore(sim, rt, events, [])
        core.law_views = [LawView(core, i, law) for i, law in law_views]
        core.name_of_event = {i: n for n, i in events.items()}
        return core

    def events_view(self):
        return _EventView(self)

    def beh_runtime(self, py_runtime):
        return NativeBehaviorRuntime(self, py_runtime)

    # ------------------------------------------------------------------ run
    def reset(self, overrides, beh_items):
        ov = []
        for bi, b in enumerate(beh_items):
            o = (overrides or {}).get(b.spec.name, {})
            bad = set(o) - {"params", "variables"}
            if bad:
                raise LobsError(f"behavior '{b.spec.name}': unknown randomize keys {sorted(bad)}")
            names = list(b.machine.variables)
            for k, v in o.get("variables", {}).items():
                if k not in names:
                    raise LobsError(f"behavior '{b.spec.name}' has no variable '{k}'")
                ov.append((bi, names.index(k), [float(v)] if np.ndim(v) == 0 else [float(x) for x in np.asarray(v, dtype=float)]))
        self.rt.reset(ov)
        # a zero-length kernel target vector keeps owned entries at zero until behaviors write them

    def observe(self, frame):
        return self.rt.observe(frame.body_pos, frame.body_quat, frame.body_linvel, frame.body_angvel,
                               frame.joint_q, frame.joint_qd, frame.joint_effort, float(frame.t))  # fmt: skip

    def step(self, user_rows):
        return self.rt.step(user_rows)
