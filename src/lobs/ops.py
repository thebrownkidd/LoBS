"""Functions for composing measures (arithmetic and comparisons are operators on Measure)."""

from __future__ import annotations

from .measure.functions import register_function  # noqa: F401
from .measure.nodes import Measure, as_measure
from .schema.measure import CustomNode, OpNode, RefNode, TimeNode


def _op(op, *ms, **args) -> Measure:
    return Measure(OpNode(op=op, inputs=[as_measure(m).node for m in ms], args=args))


def ref(name: str) -> Measure:
    return Measure(RefNode(name=name))


def time() -> Measure:
    return Measure(TimeNode())


def sin(m):
    return _op("sin", m)  # noqa: E704


def cos(m):
    return _op("cos", m)  # noqa: E704


def clip(m, lo=None, hi=None):
    return _op("clip", m, lo=lo, hi=hi)  # noqa: E704


def where(cond, a, b):
    return _op("where", cond, a, b)  # noqa: E704


def dot(a, b):
    return _op("dot", a, b)  # noqa: E704


def cross(a, b):
    return _op("cross", a, b)  # noqa: E704


def norm(m):
    return _op("norm", m)  # noqa: E704


def max_of(*ms):
    return _op("max_of", *ms)  # noqa: E704


def min_of(*ms):
    return _op("min_of", *ms)  # noqa: E704


def mean_of(*ms):
    return _op("mean_of", *ms)  # noqa: E704


def sum_of(*ms):
    return _op("sum_of", *ms)  # noqa: E704


def custom(fn: str, inputs: dict, unit: str | None = None, shape=()) -> Measure:
    """Use a function registered with ``register_function``. Only its name is serialized."""
    return Measure(
        CustomNode(
            fn=fn,
            inputs={k: as_measure(v).node for k, v in inputs.items()},
            unit=unit,
            shape=list(shape),
        )  # fmt: skip
    )
