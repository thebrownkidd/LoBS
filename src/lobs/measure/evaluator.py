"""Evaluator: runs a compiled node list on StateFrames; also builds measure metadata."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..errors import MeasureError
from ..registry import Registry
from ..state import StateFrame
from .compile import CNode, Compiler
from .context import EvalCtx
from .units import fmt


HIDDEN_PREFIXES = ("__force.", "__beh.")  # measures backing force laws and behaviors


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
        # nodes with side effects (state, RNG draws, live runtime reads) run every tick, in node order, so
        # semantics and random streams are unchanged; everything else is computed on first access
        self._eager = [
            i for i in self._order
            if self.nodes[i].stateful or (self.nodes[i].needs & {"random", "runtime"})
        ]  # fmt: skip
        self._public_names = [n for n in self.outputs if not n.startswith(HIDDEN_PREFIXES)]

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

    def evaluate(self, frame: StateFrame, rng=None, runtime=None, ctx=None) -> LazyValues:
        """Measure values for one frame. Side-effecting nodes are computed now; the rest on first access."""
        if ctx is None:
            ctx = EvalCtx(self.reg, frame, rng=rng, runtime=runtime)
        lv = LazyValues(self, ctx)
        for i in self._eager:
            lv.node(i)
        return lv

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


class LazyValues(Mapping):
    """``name -> value`` for one tick, computing pure nodes on demand (memoised)."""

    __slots__ = ("ev", "ctx", "vals", "_py")

    def __init__(self, ev: Evaluator, ctx: EvalCtx):
        self.ev, self.ctx, self.vals, self._py = ev, ctx, {}, {}

    def node(self, i: int):
        v = self.vals.get(i)
        if v is None:
            n = self.ev.nodes[i]
            v = n.fn(self.ctx, [self.node(j) for j in n.inputs], n.state)
            self.vals[i] = v
        return v

    def __getitem__(self, name: str):
        hit = self._py.get(name, _MISSING)
        if hit is _MISSING:
            i = self.ev.outputs[name]
            hit = self._py[name] = _py(self.node(i), self.ev.nodes[i])
        return hit

    def __iter__(self):
        return iter(self.ev.outputs)

    def __len__(self) -> int:
        return len(self.ev.outputs)

    def __contains__(self, name) -> bool:
        return name in self.ev.outputs


class PublicValues(Mapping):
    """Read-only view of the non-hidden measures of a ``LazyValues``."""

    __slots__ = ("lv",)

    def __init__(self, lv: LazyValues):
        self.lv = lv

    def __getitem__(self, name: str):
        if name.startswith(HIDDEN_PREFIXES):
            raise KeyError(name)
        return self.lv[name]

    def __iter__(self):
        return iter(self.lv.ev._public_names)

    def __len__(self) -> int:
        return len(self.lv.ev._public_names)

    def __contains__(self, name) -> bool:
        return name in self.lv.ev.outputs and not str(name).startswith(HIDDEN_PREFIXES)

    def __eq__(self, other) -> bool:  # value comparison (arrays included); forces evaluation
        if not isinstance(other, Mapping) or set(self) != set(other):
            return False
        return all(np.array_equal(np.asarray(self[k]), np.asarray(other[k]), equal_nan=True) for k in self)

    __hash__ = None


_MISSING = object()
