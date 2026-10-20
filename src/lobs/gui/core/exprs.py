"""Measure expressions typed in the GUI: Python text evaluated in a small, documented namespace.

This evaluates user-typed code in-process; it is meant for your own worlds on your own machine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ... import act, ops, props, rel
from ...measure.nodes import Measure, as_measure


@dataclass
class ExprError(Exception):
    message: str
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        where = f" (line {self.line}, col {self.column})" if self.line else ""
        return f"{self.message}{where}"


def namespace(world) -> dict:
    """Names available to expressions: props, rel, ops, act, np, T('path'), and every actor/anchor by name."""
    ns = {
        "props": props,
        "rel": rel,
        "ops": ops,
        "act": act,
        "np": np,
        "T": lambda p: str(p),
        "time": ops.time,
    }
    ns["world"] = world
    for a in world.spec.actors:
        if a.name.isidentifier():
            ns.setdefault(a.name, a.name)
    for an in world.spec.anchors:
        if an.name.isidentifier():
            ns.setdefault(an.name, f"world/{an.name}")
    return ns


def evaluate(text: str, world) -> Measure:
    """Evaluate ``text`` to a Measure; raises ExprError with a position when it cannot."""
    text = text.strip()
    if not text:
        raise ExprError("empty expression")
    try:
        code = compile(text, "<measure>", "eval")
    except SyntaxError as e:
        raise ExprError(e.msg or "syntax error", e.lineno, e.offset) from None
    safe = {"abs": abs, "min": min, "max": max, "float": float}
    try:
        value = eval(code, {"__builtins__": safe}, namespace(world))  # noqa: S307
        return as_measure(value)
    except Exception as e:  # noqa: BLE001 - surface any user error as a message
        raise ExprError(f"{type(e).__name__}: {e}") from None


def set_measure(spec, name: str, text: str, world) -> None:
    """Compile ``text`` and store the measure node plus its source text (for round-tripping)."""
    m = evaluate(text, world)
    spec.measures[name] = m.node
    spec.metadata.setdefault("gui", {}).setdefault("expr", {})[name] = text.strip()


def remove_measure(spec, name: str) -> None:
    spec.measures.pop(name, None)
    spec.metadata.get("gui", {}).get("expr", {}).pop(name, None)


def source_of(spec, name: str) -> str | None:
    return spec.metadata.get("gui", {}).get("expr", {}).get(name)
