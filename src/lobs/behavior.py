"""Behavior runtime: state machines and Python scripts that decide each tick and write commands.

Tick order (see docs/conventions.md):  measures -> external action -> behaviors -> force laws -> wrenches
-> backend step.  Every behavior reads one tick-start snapshot (optionally ``latency`` ticks old), so the
declaration order of behaviors never changes results. Behaviors command through channels that already
exist: actuator targets and world-frame forces (``Steer`` is acceleration-limited force), plus switches for
force laws and other behaviors. Switches take effect after all behaviors have decided.
"""

from __future__ import annotations

import zlib
from collections import deque

import numpy as np

from .errors import LobsError
from .forces import apply_force_to
from .measure.context import EvalCtx
from .schema import behavior as S

HIDDEN = "__beh."
_EPS = 1e-9

_SCRIPTS: dict[str, object] = {}
_POLICY_ACTUATES: dict[str, list[str]] = {}


def behavior(name: str | None = None):
    """Register a Python behavior: ``@lobs.behavior("chase")`` on ``f(ctx)`` or a class with
    ``reset(ctx)`` / ``step(ctx)``. Use it with ``world.add_behavior(..., script="chase")``."""

    def deco(obj):
        _SCRIPTS[name or obj.__name__] = obj
        return obj

    if callable(name):  # used bare: @lobs.behavior
        obj, name = name, None
        return deco(obj)
    return deco


def policy(name: str, fn, *, observe, actuate):
    """Register ``fn(obs) -> action`` as a behavior: ``obs`` is the flat vector of the ``observe`` measures
    (as sensed with the behavior's latency), ``action`` one value per ``actuate`` actuator."""
    observe, actuate = list(observe), list(actuate)

    def step(ctx):
        obs = np.concatenate(
            [np.atleast_1d(np.asarray(ctx.measure(m), dtype=float)).ravel() for m in observe]
        )
        act = np.atleast_1d(np.asarray(fn(obs), dtype=float)).ravel()
        if act.shape != (len(actuate),):
            raise LobsError(
                f"policy '{name}' returned {act.shape[0]} values for {len(actuate)} actuators"
            )
        for a, v in zip(actuate, act, strict=True):
            ctx.actuate(a, float(v))

    _SCRIPTS[name] = step
    _POLICY_ACTUATES[name] = actuate
    return name


behavior.policy = policy


def policy_actuators(name: str) -> list[str]:
    return list(_POLICY_ACTUATES.get(name, []))


def registered(name: str) -> bool:
    return name in _SCRIPTS


def hidden_measures(spec) -> dict:
    out = {}
    for b in spec.behaviors:
        for k, node in b.nodes().items():
            out[f"{HIDDEN}{b.name}.{k}"] = node
    return out


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.zeros(3)


def _clip_norm(v, limit):
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit and n > _EPS else v


class _Cmd:
    """Commands one behavior writes on one decision. Declarative actions and scripts share these."""

    def __init__(self, rt, b, ctx):
        self.rt, self.b, self.ctx = rt, b, ctx
        self.targets: dict[int, float] = {}
        self.forces: list = []  # (thing, vec) held until the next decision
        self.once: list = []  # (thing, vec) applied on the decision tick only
        self.hold = True  # False while running on_enter / on_exit
        self.events: set = set()
        self.force_switch: dict = {}
        self.behavior_switch: dict = {}

    # ---- resolution
    def thing(self, ref):
        if ref is None:
            if self.b.on is None:
                raise LobsError(
                    f"behavior '{self.b.spec.name}' has no actor; give 'on=' explicitly"
                )
            ref = self.b.on
        return self.rt.reg.resolve(str(getattr(ref, "path", ref)))

    # ---- commands
    def actuate(self, actuator, value):
        idx = self.rt.reg.resolve_actuator(str(getattr(actuator, "path", actuator)))
        if idx not in self.b.owned:
            raise LobsError(
                f"behavior '{self.b.spec.name}' commands actuator '{self.rt.reg.actuators[idx].path}' "
                f"without owning it (list it in the script's actuates=)"
            )
        value = float(value)
        if not np.isfinite(value):
            raise LobsError(f"behavior '{self.b.spec.name}': non-finite actuator command {value}")
        self.targets[idx] = value

    def force(self, on, vector, max_force=None):
        v = np.asarray(vector, dtype=float).reshape(-1)
        if v.shape != (3,) or not np.all(np.isfinite(v)):
            raise LobsError(
                f"behavior '{self.b.spec.name}': force must be a finite 3-vector, got {vector}"
            )
        if max_force is not None:
            v = _clip_norm(v, float(max_force))
        (self.forces if self.hold else self.once).append((self.thing(on), v))

    def steer(self, target=None, mode="seek", on=None, speed=1.0, max_accel=5.0, gain=4.0, plane=True,
              stop_radius=0.0):  # fmt: skip
        me = self.thing(on)
        ctx = self.ctx
        mass = self.rt.mass_of(me)
        p, v = ctx.pose(me)[0].copy(), ctx.twist(me)[0].copy()
        if mode == "hold":
            v_des, d = np.zeros(3), np.zeros(3)
        else:
            if target is None:
                raise LobsError(f"steer mode '{mode}' needs a target")
            tp = self.thing(target)
            pt, vt = ctx.pose(tp)[0].copy(), ctx.twist(tp)[0].copy()
            if plane:
                pt[2], vt[2] = p[2], 0.0
            d = pt - p
            dist = float(np.linalg.norm(d))
            if mode == "seek":
                v_des = speed * _unit(d)
            elif mode == "flee":
                v_des = -speed * _unit(d)
            elif mode == "arrive":
                room = max(dist - stop_radius, 0.0)
                v_des = min(speed, 0.8 * np.sqrt(2.0 * max_accel * room)) * _unit(d)
            elif mode == "intercept":
                horizon = dist / speed if speed > _EPS else 0.0
                v_des = speed * _unit(pt + vt * horizon - p)
            else:
                raise LobsError(f"unknown steer mode '{mode}'")
        if plane:
            v_des[2], v[2] = 0.0, 0.0
        acc = _clip_norm(gain * (v_des - v), max_accel)
        if plane:
            acc[2] = 0.0
        (self.forces if self.hold else self.once).append((me, mass * acc))

    def set_var(self, name, value):
        m = self.b.spec.machine
        if m is None or name not in m.variables:
            raise LobsError(f"behavior '{self.b.spec.name}' has no variable '{name}'")
        arr = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise LobsError(
                f"behavior '{self.b.spec.name}': non-finite value for variable '{name}'"
            )
        want = () if isinstance(m.variables[name], (int, float)) else (3,)
        if arr.shape != want:
            raise LobsError(f"variable '{name}' has shape {want}, got {arr.shape}")
        self.b.vars[name] = float(arr) if want == () else arr.copy()

    def emit(self, event):
        self.events.add(str(event))

    def enable_force(self, force, enabled=True):
        self.force_switch[str(force)] = bool(enabled)

    def enable_behavior(self, behavior, enabled=True):
        self.behavior_switch[str(behavior)] = bool(enabled)


class BehaviorContext:
    """What a Python behavior sees and may command (also the object passed to ``reset``)."""

    def __init__(self, cmd: _Cmd, b, snap_values, snap_frame, t, dt, tick):
        self._cmd, self._b, self._values, self._frame = cmd, b, snap_values, snap_frame
        self.t, self.dt, self.tick = t, dt, tick
        self.name = b.spec.name
        self.rng = b.rng
        self.params = b.params
        self.state = b.script_state

    def measure(self, name):
        """A named (public) measure as sensed with this behavior's latency."""
        if name not in self._values:
            raise LobsError(f"behavior '{self.name}': unknown measure '{name}'")
        return self._values[name]

    def pos(self, thing):
        return self._cmd.ctx.pose(self._cmd.thing(thing))[0].copy()

    def vel(self, thing):
        return self._cmd.ctx.twist(self._cmd.thing(thing))[0].copy()

    def steer(self, to=None, **kw):
        return self._cmd.steer(target=to, **kw)

    def actuate(self, actuator, value):
        return self._cmd.actuate(actuator, value)

    def force(self, on, vector, max_force=None):
        return self._cmd.force(on, vector, max_force)

    def emit(self, event):
        return self._cmd.emit(event)

    def enable_force(self, force, enabled=True):
        return self._cmd.enable_force(force, enabled)

    def enable_behavior(self, behavior, enabled=True):
        return self._cmd.enable_behavior(behavior, enabled)


class _B:
    """Run-time state of one behavior (exposed to measures through ``runtime.behaviors``)."""

    def __init__(self, spec, on, owned):
        self.spec, self.on, self.owned = spec, on, owned
        self.machine = spec.machine
        self.state_names = [s.name for s in self.machine.states] if self.machine else []
        self.script_obj = None
        self.reset_state()

    def reset_state(self):
        m = self.machine
        self.enabled = self.spec.enabled
        self.state_name = (m.initial or self.state_names[0]) if m else ""
        self.state_idx = self.state_names.index(self.state_name) if m else 0
        self.enter_t = 0.0
        self.entered = False
        self.vars = (
            {
                k: (float(v) if isinstance(v, (int, float)) else np.array(v, dtype=float))
                for k, v in m.variables.items()
            }
            if m
            else {}
        )
        self.params = dict(self.spec.script.params) if self.spec.script else {}
        self.script_state: dict = {}
        self.rng = np.random.default_rng(0)
        self.targets: dict[int, float] = {}
        self.forces: list = []

    def goto(self, name, t):
        self.state_name, self.state_idx, self.enter_t = name, self.state_names.index(name), t


class BehaviorRuntime:
    def __init__(self, reg, spec, laws, tick_dt):
        self.reg, self.tick_dt = reg, tick_dt
        self.laws = {law.name: law for law in laws}
        self.events: set = set()
        self.log: list = []  # (t, behavior, from, to)
        owners: dict[int, str] = {}
        self.items: list[_B] = []
        for bs in spec.behaviors:
            on = None if bs.on is None else reg.resolve(bs.on)
            owned = set()
            for a in bs.owned_actuators():
                i = reg.resolve_actuator(a)
                if i in owners and owners[i] != bs.name:
                    raise LobsError(
                        f"actuator '{reg.actuators[i].path}' is owned by both '{owners[i]}' and '{bs.name}'"
                    )
                owners[i] = bs.name
                owned.add(i)
            b = _B(bs, on, owned)
            if bs.script is not None:
                if bs.script.fn not in _SCRIPTS:
                    raise LobsError(
                        f"behavior '{bs.name}': script '{bs.script.fn}' is not registered (use @lobs.behavior)"
                    )
                obj = _SCRIPTS[bs.script.fn]
                b.script_obj = obj() if isinstance(obj, type) else obj
            self.items.append(b)
        self.by_name = {b.spec.name: b for b in self.items}
        self.owned = sorted(owners)
        self.period = [max(1, round(b.spec.period / tick_dt)) for b in self.items]
        self.latency = [round(b.spec.latency / tick_dt) for b in self.items]
        self.buffer: deque = deque(maxlen=max(self.latency, default=0) + 1)
        self.names = {b.spec.name: b for b in self.items}

    # exposed to measures (props.behavior_*, force_*, ops.event)
    @property
    def behaviors(self):
        return self.names

    @property
    def forces(self):
        return self.laws

    def mass_of(self, thing) -> float:
        if thing.kind == "actor":
            idx = [i for i in self.reg.actors[thing.index].bodies if not self.reg.bodies[i].static]
        elif thing.kind == "body":
            idx = [thing.index] if not self.reg.bodies[thing.index].static else []
        else:
            idx = []
        m = sum(self.reg.bodies[i].mass for i in idx)
        if m <= 0:
            raise LobsError(f"cannot steer '{thing.path}': it is static or has no mass")
        return m

    # ------------------------------------------------------------ lifecycle
    def reset(self, rng, overrides=None):
        base = int(rng.integers(2**32)) if self.items else 0
        self.events, self.log = set(), []
        self.buffer.clear()
        for b in self.items:
            b.reset_state()
            b.rng = np.random.default_rng([base, zlib.crc32(b.spec.name.encode())])
            ov = (overrides or {}).get(b.spec.name, {})
            bad = set(ov) - {"params", "variables"}
            if bad:
                raise LobsError(f"behavior '{b.spec.name}': unknown randomize keys {sorted(bad)}")
            b.params.update(ov.get("params", {}))
            for k, v in ov.get("variables", {}).items():
                if k not in b.vars:
                    raise LobsError(f"behavior '{b.spec.name}' has no variable '{k}'")
                b.vars[k] = float(v) if np.ndim(v) == 0 else np.asarray(v, dtype=float)

    # ------------------------------------------------------------ one tick
    def step(self, tick, t, frame, values, add_wrench, now_ctx) -> dict:
        """Run all behaviors; returns ``{actuator_index: target}`` set this tick (held targets included)."""
        self.buffer.append((frame, values))
        emitted: set = set()
        force_sw, beh_sw = {}, {}
        for bi, b in enumerate(self.items):
            if not b.enabled:
                b.forces = []
                continue
            if tick % self.period[bi] == 0:
                lat = self.latency[bi]
                snap_frame, snap_values = (
                    self.buffer[-1 - lat] if len(self.buffer) > lat else self.buffer[0]
                )
                cmd = _Cmd(self, b, EvalCtx(self.reg, snap_frame))
                if b.spec.machine is not None:
                    self._decide_machine(b, cmd, t, snap_values)
                else:
                    self._decide_script(b, cmd, t, tick, snap_values, snap_frame)
                b.targets.update(cmd.targets)
                b.forces = cmd.forces
                for thing, vec in cmd.once:
                    apply_force_to(self.reg, now_ctx, thing, vec, add_wrench)
                emitted |= cmd.events
                force_sw.update(cmd.force_switch)
                beh_sw.update(cmd.behavior_switch)
            for thing, vec in b.forces:
                apply_force_to(self.reg, now_ctx, thing, vec, add_wrench)
        for name, on in force_sw.items():
            if name not in self.laws:
                raise LobsError(f"enable_force: unknown force '{name}'")
            self.laws[name].enabled = on
        for name, on in beh_sw.items():
            if name not in self.by_name:
                raise LobsError(f"enable_behavior: unknown behavior '{name}'")
            self.by_name[name].enabled = on
        self.events = emitted
        out = {}
        for b in self.items:
            out.update(b.targets)
        return out

    # ------------------------------------------------------------ state machines
    def _run(self, b, cmd, phase_actions, prefix, vals, hold):
        cmd.hold = hold
        for i, a in enumerate(phase_actions):
            key = f"{HIDDEN}{b.spec.name}.{prefix}.{i}"
            if isinstance(a, S.Actuate):
                cmd.actuate(a.actuator, float(vals[f"{key}.value"]))
            elif isinstance(a, S.Force):
                cmd.force(a.on, vals[f"{key}.vector"], a.max_force)
            elif isinstance(a, S.Steer):
                cmd.steer(
                    a.target, a.mode, a.on, a.speed, a.max_accel, a.gain, a.plane, a.stop_radius
                )
            elif isinstance(a, S.SetVar):
                cmd.set_var(a.name, vals[f"{key}.value"])
            elif isinstance(a, S.Emit):
                cmd.emit(a.event)
            elif isinstance(a, S.EnableForce):
                cmd.enable_force(a.force, a.enabled)
            elif isinstance(a, S.EnableBehavior):
                cmd.enable_behavior(a.behavior, a.enabled)
        cmd.hold = True

    def _decide_machine(self, b, cmd, t, vals):
        m = b.machine
        states = {s.name: s for s in m.states}
        if not b.entered:
            b.entered = True
            b.enter_t = t
            self._run(
                b, cmd, states[b.state_name].on_enter, f"{b.state_name}.on_enter", vals, False
            )
        cands = sorted(
            (i for i, tr in enumerate(m.transitions) if tr.frm == b.state_name),
            key=lambda i: (-m.transitions[i].priority, i),
        )
        for i in cands:
            tr = m.transitions[i]
            if t - b.enter_t >= tr.min_time - _EPS and bool(
                vals[f"{HIDDEN}{b.spec.name}.t{i}.when"]
            ):
                self._run(
                    b, cmd, states[b.state_name].on_exit, f"{b.state_name}.on_exit", vals, False
                )
                self.log.append((t, b.spec.name, b.state_name, tr.to))
                b.goto(tr.to, t)
                self._run(b, cmd, states[tr.to].on_enter, f"{tr.to}.on_enter", vals, False)
                break
        self._run(b, cmd, states[b.state_name].do, f"{b.state_name}.do", vals, True)

    # ------------------------------------------------------------ scripts
    def _decide_script(self, b, cmd, t, tick, vals, frame):
        ctx = BehaviorContext(cmd, b, vals, frame, t, self.tick_dt, tick)
        obj = b.script_obj
        if not b.entered:
            b.entered = True
            if hasattr(obj, "reset"):
                obj.reset(ctx)
        fn = obj.step if hasattr(obj, "step") else obj
        if not callable(fn):
            raise LobsError(f"behavior '{b.spec.name}': script has no step(ctx)")
        fn(ctx)

    def states(self) -> dict:
        return {b.spec.name: b.state_idx for b in self.items}


__all__ = [
    "BehaviorContext",
    "BehaviorRuntime",
    "behavior",
    "hidden_measures",
    "policy",
]
