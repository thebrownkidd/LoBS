"""Force laws: measure-valued, optionally triggered, vector-valued and budgeted forces.

Every law is evaluated once per tick from the state at the start of the tick and applied as a
zero-order-hold wrench over the substeps. The magnitude / vector / trigger of a law are compiled as
hidden measures (``__force.<name>``, ``.vec``, ``.when``) in the simulation's evaluator, so they can use
any property, relative measure, operator or trigger.

Budgets keep adversarial forces bounded and fair: ``peak`` clamps ``|F|``, ``impulse`` caps the total
``integral |F| dt`` per episode (the last tick is scaled so it is exhausted exactly), ``active_time`` caps
the seconds a law may be active (tick granularity). All state resets with the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import math3d as m3
from .errors import LobsError
from .measure.catalog import parse_axis
from .measure.context import EvalCtx

HIDDEN = "__force."
_EPS = 1e-12


def hidden_measures(spec) -> dict:
    """Hidden measures for every law: magnitude (``__force.<n>``), ``.vec`` and ``.when``."""
    out = {}
    for f in spec.forces:
        if f.magnitude is not None:
            out[f"{HIDDEN}{f.name}"] = f.magnitude
        if f.vector is not None:
            out[f"{HIDDEN}{f.name}.vec"] = f.vector
        if f.when is not None:
            out[f"{HIDDEN}{f.name}.when"] = f.when
    return out


@dataclass
class Law:
    name: str
    on: object  # Thing
    toward: object | None
    direction: np.ndarray | None
    axis: tuple | None  # (Thing, local unit vector)
    reaction: bool
    has_vector: bool = False
    has_when: bool = False
    peak_limit: float | None = None
    impulse_limit: float | None = None
    active_time_limit: float | None = None
    adversary: bool = False
    enabled: bool = True  # behaviors can switch a law off/on at run time
    # ---- per-episode state
    impulse: float = 0.0
    active_time: float = 0.0
    peak: float = 0.0
    active: bool = False
    last_force: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def reset(self) -> None:
        self.enabled = True
        self.impulse = self.active_time = self.peak = 0.0
        self.active = False
        self.last_force = np.zeros(3)

    def stats(self) -> dict:
        return {
            "impulse": self.impulse,
            "active_time": self.active_time,
            "peak": self.peak,
            "last_force": self.last_force.copy(),
            "active": self.active,
        }


def compile_laws(reg, spec) -> list[Law]:
    laws, seen = [], set()
    for f in spec.forces:
        if f.name in seen:
            raise LobsError(f"duplicate force law '{f.name}'")
        seen.add(f.name)
        given = [x for x in (f.toward, f.direction, f.axis, f.vector) if x is not None]
        if len(given) != 1:
            raise LobsError(
                f"force '{f.name}': give exactly one of toward=, direction=, axis=, vector="
            )
        if f.vector is not None and f.magnitude is not None:
            raise LobsError(f"force '{f.name}': a vector force has no separate magnitude")
        if f.vector is None and f.magnitude is None:
            raise LobsError(f"force '{f.name}': give a magnitude (or a vector=)")
        try:
            on = reg.resolve(f.on)
            toward = reg.resolve(f.toward) if f.toward is not None else None
            axis = None
            if f.axis is not None:
                axis = (reg.resolve(f.axis[0]), parse_axis(f.axis[1])[0])
        except LobsError as e:
            raise LobsError(f"force '{f.name}': {e}") from None
        if on.kind not in ("actor", "body", "anchor"):
            raise LobsError(f"force '{f.name}': cannot apply a force to a {on.kind}")
        if on.kind == "anchor" and reg.anchors[on.index].parent < 0:
            raise LobsError(
                f"force '{f.name}': '{f.on}' is a world anchor (fixed); attach the force to a body or actor"
            )
        direction = None
        if f.direction is not None:
            direction = np.asarray(f.direction, dtype=float)
            if np.linalg.norm(direction) < 1e-12:
                raise LobsError(f"force '{f.name}': direction must be non-zero")
            direction = direction / np.linalg.norm(direction)
        b = f.budget
        for label, val in (
            ("peak", b.peak if b else None),
            ("impulse", b.impulse if b else None),
            ("active_time", b.active_time if b else None),
        ):
            if val is not None and not val >= 0:
                raise LobsError(f"force '{f.name}': budget {label} must be >= 0")
        laws.append(
            Law(
                f.name,
                on,
                toward,
                direction,
                axis,
                f.reaction and f.toward is not None,
                has_vector=f.vector is not None,
                has_when=f.when is not None,
                peak_limit=b.peak if b else None,
                impulse_limit=b.impulse if b else None,
                active_time_limit=b.active_time if b else None,
                adversary=f.adversary,
            )  # fmt: skip
        )
    return laws


def _apply(reg, ctx: EvalCtx, thing, force: np.ndarray, add_wrench) -> None:
    zero = np.zeros(3)
    if thing.kind == "body":
        b = reg.bodies[thing.index]
        if not b.static:
            add_wrench(thing.index, force, zero, ctx.body_com(thing.index)[0])
    elif thing.kind == "actor":
        bodies = [
            i
            for i in reg.actors[thing.index].bodies
            if not reg.bodies[i].static and reg.bodies[i].mass > 0
        ]
        total = sum(reg.bodies[i].mass for i in bodies)
        for i in bodies:  # share by mass: net force F at the actor's centre, no extra net torque
            add_wrench(i, force * reg.bodies[i].mass / total, zero, ctx.body_com(i)[0])
    elif thing.kind == "anchor":
        parent = reg.anchors[thing.index].parent
        if parent >= 0 and not reg.bodies[parent].static:
            add_wrench(parent, force, zero, ctx.pose(thing)[0])
    # world: nothing to push


def apply_force_to(reg, ctx: EvalCtx, thing, force: np.ndarray, add_wrench) -> None:
    """Apply a world-frame force to an actor / body / anchor (shared with behaviors)."""
    _apply(reg, ctx, thing, force, add_wrench)


def _limit(law: Law, force: np.ndarray, dt: float) -> np.ndarray:
    """Apply the budget: peak clamp, active-time cap, impulse exhaustion (exact)."""
    if law.peak_limit is not None:
        n = float(np.linalg.norm(force))
        if n > law.peak_limit and n > _EPS:
            force = force * (law.peak_limit / n)
    if law.active_time_limit is not None and law.active_time >= law.active_time_limit - 1e-9:
        return np.zeros(3)
    if law.impulse_limit is not None:
        remaining = law.impulse_limit - law.impulse
        if remaining <= _EPS:
            return np.zeros(3)
        need = float(np.linalg.norm(force)) * dt
        if need > remaining:
            force = force * (remaining / need)
    return force


def apply_laws(
    laws: list[Law], reg, frame, values: dict, add_wrench, dt: float = 0.0, ctx=None
) -> None:
    if not laws:
        return
    if ctx is None:
        ctx = EvalCtx(reg, frame)
    for law in laws:
        key = f"{HIDDEN}{law.name}"
        on = law.enabled and (not law.has_when or bool(values[f"{key}.when"]))
        if not on:
            law.active, law.last_force = False, np.zeros(3)
            continue
        if law.has_vector:
            force = np.asarray(values[f"{key}.vec"], dtype=float)
            if not np.all(np.isfinite(force)):
                raise LobsError(
                    f"force '{law.name}' produced a non-finite vector ({force}) at t={frame.t:g}"
                )
        else:
            mag = float(values[key])
            if not np.isfinite(mag):
                raise LobsError(
                    f"force '{law.name}' produced a non-finite magnitude ({mag}) at t={frame.t:g}"
                )
            if law.toward is not None:
                d = ctx.pose(law.toward)[0] - ctx.pose(law.on)[0]
                n = np.linalg.norm(d)
                if n < 1e-12:
                    law.active, law.last_force = False, np.zeros(3)
                    continue  # coincident points: direction undefined, apply nothing
                unit = d / n
            elif law.direction is not None:
                unit = law.direction
            else:
                thing, local = law.axis
                unit = ctx.pose(thing)[1] @ local
            force = mag * unit
        force = _limit(law, force, dt)
        fmag = float(np.linalg.norm(force))
        if (
            fmag == 0.0
        ):  # nothing delivered (zero vector or budget exhausted): not active, costs nothing
            law.active, law.last_force = False, force
            continue
        law.active, law.last_force = True, force
        law.impulse += fmag * dt
        law.active_time += dt
        law.peak = max(law.peak, fmag)
        _apply(reg, ctx, law.on, force, add_wrench)
        if law.reaction:
            _apply(reg, ctx, law.toward, -force, add_wrench)


__all__ = ["HIDDEN", "Law", "apply_force_to", "apply_laws", "compile_laws", "hidden_measures", "m3"]
