"""Force laws: measure-valued forces applied every tick (zero-order hold over the substeps).

The magnitude of every law is compiled as a hidden measure (``__force.<name>``) inside the simulation's
evaluator, so it can use any property / relative measure / operator, including time dependence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import math3d as m3
from .errors import LobsError
from .measure.catalog import parse_axis
from .measure.context import EvalCtx

HIDDEN = "__force."


def hidden_measures(spec) -> dict:
    return {f"{HIDDEN}{f.name}": f.magnitude for f in spec.forces}


@dataclass
class Law:
    name: str
    on: object  # Thing
    toward: object | None
    direction: np.ndarray | None
    axis: tuple | None  # (Thing, local unit vector)
    reaction: bool


def compile_laws(reg, spec) -> list[Law]:
    laws, seen = [], set()
    for f in spec.forces:
        if f.name in seen:
            raise LobsError(f"duplicate force law '{f.name}'")
        seen.add(f.name)
        given = [x for x in (f.toward, f.direction, f.axis) if x is not None]
        if len(given) != 1:
            raise LobsError(f"force '{f.name}': give exactly one of toward=, direction=, axis=")
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
        laws.append(Law(f.name, on, toward, direction, axis, f.reaction and f.toward is not None))
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
        for i in (
            bodies
        ):  # share the force by mass: net force F at the actor's centre, no extra net torque
            add_wrench(i, force * reg.bodies[i].mass / total, zero, ctx.body_com(i)[0])
    elif thing.kind == "anchor":
        parent = reg.anchors[thing.index].parent
        if parent >= 0 and not reg.bodies[parent].static:
            add_wrench(parent, force, zero, ctx.pose(thing)[0])
    # world: nothing to push


def apply_laws(laws: list[Law], reg, frame, values: dict, add_wrench) -> None:
    if not laws:
        return
    ctx = EvalCtx(reg, frame)
    for law in laws:
        mag = float(values[f"{HIDDEN}{law.name}"])
        if not np.isfinite(mag):
            raise LobsError(
                f"force '{law.name}' produced a non-finite magnitude ({mag}) at t={frame.t:g}"
            )
        if law.toward is not None:
            d = ctx.pose(law.toward)[0] - ctx.pose(law.on)[0]
            n = np.linalg.norm(d)
            if n < 1e-12:
                continue  # coincident points: direction undefined, apply nothing
            unit = d / n
        elif law.direction is not None:
            unit = law.direction
        else:
            thing, local = law.axis
            unit = ctx.pose(thing)[1] @ local
        force = mag * unit
        _apply(reg, ctx, law.on, force, add_wrench)
        if law.reaction:
            _apply(reg, ctx, law.toward, -force, add_wrench)


__all__ = ["HIDDEN", "Law", "apply_laws", "compile_laws", "hidden_measures", "m3"]
