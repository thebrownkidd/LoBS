"""Evaluator: runs a compiled node list on StateFrames; also builds measure metadata."""

from __future__ import annotations

import numpy as np

from ..errors import MeasureError
from ..registry import Registry
from ..state import StateFrame
from .compile import CNode, Compiler
from .context import EvalCtx
from .units import fmt


class Evaluator:
    def __init__(
        self, reg: Registry, nodes: list[CNode], outputs: dict[str, int], definitions: dict
    ):
        self.reg, self.nodes, self.outputs, self.definitions = reg, nodes, outputs, definitions
        used = self._used()
        self.needs = (
            frozenset().union(*(self.nodes[i].needs for i in used)) if used else frozenset()
        )
        self.caps = frozenset().union(*(self.nodes[i].caps for i in used)) if used else frozenset()
        self._order = sorted(used)

    def _used(self) -> set[int]:
        seen, stack = set(), list(self.outputs.values())
        while stack:
            i = stack.pop()
            if i not in seen:
                seen.add(i)
                stack.extend(self.nodes[i].inputs)
        return seen

    def reset(self) -> None:
        for n in self.nodes:
            n.state.clear()

    def evaluate(self, frame: StateFrame, rng=None, runtime=None, ctx=None) -> dict:
        if ctx is None:
            ctx = EvalCtx(self.reg, frame, rng=rng, runtime=runtime)
        vals: dict[int, np.ndarray] = {}
        for i in self._order:
            n = self.nodes[i]
            vals[i] = n.fn(ctx, [vals[j] for j in n.inputs], n.state)
        return {name: _py(vals[i], self.nodes[i]) for name, i in self.outputs.items()}

    def meta(self) -> dict:
        out = {}
        for name, i in self.outputs.items():
            n = self.nodes[i]
            out[name] = {
                "unit": fmt(n.unit) if n.unit is not None else "dimensionless",
                "shape": list(n.shape),
                "kind": n.kind,
                "provenance": n.prov,
                "definition": self.definitions.get(name),
            }
        return out


def _py(v, node: CNode):
    if node.kind == "bool":
        a = np.asarray(v, dtype=bool)
        return bool(a) if a.shape == () else a
    a = np.asarray(v, dtype=float)
    return float(a) if a.shape == () else a


def build_evaluator(reg: Registry, named: dict, extra: dict | None = None,
                    only: list[str] | None = None) -> Evaluator:  # fmt: skip
    """Compile all named measures (+ optional ad-hoc ones). Raises MeasureError listing every failure."""
    all_named = dict(named)
    if extra:
        all_named.update(extra)
    comp = Compiler(reg, all_named)
    errors, outputs, defs = [], {}, {}
    for name in only if only is not None else all_named:
        try:
            outputs[name] = comp.compile_named(name)
            defs[name] = all_named[name].model_dump(mode="json")
        except MeasureError as e:
            errors.append(str(e))
    if errors:
        raise MeasureError("; ".join(errors))
    return Evaluator(reg, comp.nodes, outputs, defs)
