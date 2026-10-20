"""Behaviors: serializable state machines (and named Python scripts) that decide and command.

A behavior is not a force law: it owns state, decides every ``period`` seconds from what it could sense
``latency`` seconds ago, and *writes commands* (actuator targets, forces, force-law switches). Every
measure used by an action or transition is compiled as a hidden measure (``__beh.<name>.<key>``).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from .common import Model, Vec3
from .measure import MeasureNode


class Actuate(Model):
    do: Literal["actuate"] = "actuate"
    actuator: str
    value: MeasureNode


class Force(Model):
    do: Literal["force"] = "force"
    on: str | None = None  # default: the behavior's actor
    vector: MeasureNode  # shape (3,), world frame, newtons
    max_force: float | None = None


class Steer(Model):
    """Acceleration-limited locomotion: F = m * clip_norm(gain * (v_des - v), max_accel)."""

    do: Literal["steer"] = "steer"
    mode: Literal["seek", "arrive", "flee", "intercept", "hold"] = "seek"
    target: str | None = (
        None  # a thing with a position (actor / body / anchor); not needed for hold
    )
    on: str | None = None  # default: the behavior's actor
    speed: float = 1.0
    max_accel: float = 5.0
    gain: float = 4.0
    plane: bool = True  # steer in the world xy plane only
    stop_radius: float = 0.0  # arrive: stop this far from the target


class SetVar(Model):
    do: Literal["set_var"] = "set_var"
    name: str
    value: MeasureNode


class Emit(Model):
    do: Literal["emit"] = "emit"
    event: str


class EnableForce(Model):
    do: Literal["enable_force"] = "enable_force"
    force: str
    enabled: bool = True


class EnableBehavior(Model):
    do: Literal["enable_behavior"] = "enable_behavior"
    behavior: str
    enabled: bool = True


Action = Annotated[
    Actuate | Force | Steer | SetVar | Emit | EnableForce | EnableBehavior,
    Field(discriminator="do"),
]


class StateSpec(Model):
    name: str
    on_enter: list[Action] = Field(default_factory=list)
    do: list[Action] = Field(default_factory=list)
    on_exit: list[Action] = Field(default_factory=list)


class TransitionSpec(Model):
    frm: str
    to: str
    when: MeasureNode  # bool
    min_time: float = 0.0  # seconds the machine must have been in `frm`
    priority: int = 0  # higher first, then declaration order


class StateMachineSpec(Model):
    variables: dict[str, float | Vec3] = Field(default_factory=dict)
    initial: str | None = None
    states: list[StateSpec] = Field(default_factory=list)
    transitions: list[TransitionSpec] = Field(default_factory=list)


class ScriptSpec(Model):
    fn: str  # name registered with @lobs.behavior
    params: dict = Field(default_factory=dict)
    actuates: list[str] = Field(
        default_factory=list
    )  # actuators the script may command (ownership)


class BehaviorSpec(Model):
    name: str
    on: str | None = None  # the actor it controls (None: a global behavior)
    enabled: bool = True
    latency: float = 0.0  # seconds of sensing delay
    period: float = 0.0  # seconds between decisions (0: every tick); commands are held in between
    machine: StateMachineSpec | None = None
    script: ScriptSpec | None = None
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_body(self):
        if (self.machine is None) == (self.script is None):
            raise ValueError(
                f"behavior '{self.name}': give exactly one of a state machine or a script"
            )
        if self.latency < 0 or self.period < 0:
            raise ValueError(f"behavior '{self.name}': latency and period must be >= 0")
        return self

    # ---- helpers shared by the compiler, validation and include
    def actions(self):
        """Yield ``(key, action)`` for every action of a state machine; keys are stable and unique."""
        if self.machine is None:
            return
        for s in self.machine.states:
            for phase in ("on_enter", "do", "on_exit"):
                for i, a in enumerate(getattr(s, phase)):
                    yield f"{s.name}.{phase}.{i}", a

    def nodes(self) -> dict[str, MeasureNode]:
        """Every measure the behavior needs: ``{key: node}`` (compiled as hidden measures)."""
        out = {}
        for key, a in self.actions():
            if isinstance(a, (Actuate, SetVar)):
                out[f"{key}.value"] = a.value
            elif isinstance(a, Force):
                out[f"{key}.vector"] = a.vector
        if self.machine is not None:
            for i, t in enumerate(self.machine.transitions):
                out[f"t{i}.when"] = t.when
        return out

    def owned_actuators(self) -> list[str]:
        own = [a.actuator for _, a in self.actions() if isinstance(a, Actuate)]
        if self.script is not None:
            own += list(self.script.actuates)
        return own
