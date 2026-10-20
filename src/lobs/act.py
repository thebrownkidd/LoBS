"""Action constructors and the StateMachine builder for behaviors.

sm = lobs.StateMachine("guard_then_chase")
sm.state("guard", do=[act.steer(to=zone, mode="arrive", speed=2.0)])
sm.state("chase", do=[act.steer(to=ball, mode="intercept", speed=5.0, max_accel=9.0)])
sm.transition("guard", "chase", when=rel.between(me, ball).distance() < 4.0, min_time=0.2)
world.add_behavior("defender", on=defender, spec=sm, latency=0.15)
"""

from __future__ import annotations

from .errors import LobsError
from .measure.nodes import as_measure
from .schema import behavior as S


def _path(x):
    return None if x is None else str(getattr(x, "path", x))


def actuate(actuator, value) -> S.Actuate:
    return S.Actuate(actuator=_path(actuator), value=as_measure(value).node)


def force(vector, on=None, max_force=None) -> S.Force:
    return S.Force(on=_path(on), vector=as_measure(vector).node, max_force=max_force)


def steer(
    to=None, mode="seek", on=None, speed=1.0, max_accel=5.0, gain=4.0, plane=True, stop_radius=0.0
) -> S.Steer:
    return S.Steer(mode=mode, target=_path(to), on=_path(on), speed=speed, max_accel=max_accel, gain=gain,
                   plane=plane, stop_radius=stop_radius)  # fmt: skip


def set_var(name, value) -> S.SetVar:
    return S.SetVar(name=name, value=as_measure(value).node)


def emit(event) -> S.Emit:
    return S.Emit(event=str(event))


def enable_force(force, enabled=True) -> S.EnableForce:
    return S.EnableForce(force=_path(force), enabled=enabled)


def enable_behavior(behavior, enabled=True) -> S.EnableBehavior:
    return S.EnableBehavior(behavior=_path(behavior), enabled=enabled)


class StateMachine:
    """Builder for a declarative state machine; pass it to ``world.add_behavior(spec=...)``."""

    def __init__(self, name: str = "machine"):
        self.name = name
        self._m = S.StateMachineSpec()

    def var(self, name, initial=0.0):
        """A persistent variable (number or 3-vector), changed with ``act.set_var``."""
        self._m.variables[name] = (
            initial if isinstance(initial, (int, float)) else tuple(float(x) for x in initial)
        )
        return self

    def state(self, name, do=(), on_enter=(), on_exit=()):
        if any(s.name == name for s in self._m.states):
            raise LobsError(f"state machine '{self.name}': duplicate state '{name}'")
        self._m.states.append(
            S.StateSpec(name=name, do=list(do), on_enter=list(on_enter), on_exit=list(on_exit))
        )
        return self

    def initial(self, name):
        self._m.initial = name
        return self

    def transition(self, frm, to, when, min_time=0.0, priority=0):
        self._m.transitions.append(
            S.TransitionSpec(
                frm=frm, to=to, when=as_measure(when).node, min_time=min_time, priority=priority
            )
        )
        return self

    def spec(self) -> S.StateMachineSpec:
        return self._m.model_copy(deep=True)
