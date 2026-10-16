"""User-facing ``Measure``: a thin wrapper around a MeasureNode with operator overloads.

Not a pydantic model (so ``==`` etc. can be overloaded without breaking validation).
"""

from __future__ import annotations

import numpy as np

from ..errors import MeasureError
from ..schema.measure import ConstNode, OpNode


class Measure:
    __slots__ = ("node",)
    __hash__ = object.__hash__
    __array_ufunc__ = None  # make numpy defer to our operators (np.float64 * Measure)

    def __init__(self, node):
        self.node = node

    def __repr__(self) -> str:
        return f"Measure({self.node.model_dump_json()})"

    def __bool__(self):
        raise TypeError("a Measure has no truth value; use '&', '|', '~' for logic")

    # ---- arithmetic
    def _bin(self, op, other, swap=False):
        a, b = (as_measure(other), self) if swap else (self, as_measure(other))
        return Measure(OpNode(op=op, inputs=[a.node, b.node]))

    def __add__(self, o):
        return self._bin("add", o)  # noqa: E704

    def __radd__(self, o):
        return self._bin("add", o, True)  # noqa: E704

    def __sub__(self, o):
        return self._bin("sub", o)  # noqa: E704

    def __rsub__(self, o):
        return self._bin("sub", o, True)  # noqa: E704

    def __mul__(self, o):
        return self._bin("mul", o)  # noqa: E704

    def __rmul__(self, o):
        return self._bin("mul", o, True)  # noqa: E704

    def __truediv__(self, o):
        return self._bin("div", o)  # noqa: E704

    def __rtruediv__(self, o):
        return self._bin("div", o, True)  # noqa: E704

    def __pow__(self, n):
        if not isinstance(n, (int, float)) or isinstance(n, bool):
            raise MeasureError("exponent must be a plain number")
        return Measure(OpNode(op="pow", inputs=[self.node], args={"n": float(n)}))

    def __neg__(self):
        return Measure(OpNode(op="neg", inputs=[self.node]))  # noqa: E704

    def __abs__(self):
        return Measure(OpNode(op="abs", inputs=[self.node]))  # noqa: E704

    # ---- comparisons / logic (note: == is NOT overloaded; use .eq)
    def __lt__(self, o):
        return self._bin("lt", o)  # noqa: E704

    def __le__(self, o):
        return self._bin("le", o)  # noqa: E704

    def __gt__(self, o):
        return self._bin("gt", o)  # noqa: E704

    def __ge__(self, o):
        return self._bin("ge", o)  # noqa: E704

    def __and__(self, o):
        return self._bin("and", o)  # noqa: E704

    def __rand__(self, o):
        return self._bin("and", o, True)  # noqa: E704

    def __or__(self, o):
        return self._bin("or", o)  # noqa: E704

    def __ror__(self, o):
        return self._bin("or", o, True)  # noqa: E704

    def __invert__(self):
        return Measure(OpNode(op="not", inputs=[self.node]))  # noqa: E704

    def eq(self, other, tol: float = 0.0):
        o = as_measure(other)
        return Measure(OpNode(op="eq", inputs=[self.node, o.node], args={"tol": float(tol)}))

    # ---- vectors
    def __getitem__(self, i: int):
        return Measure(OpNode(op="index", inputs=[self.node], args={"i": int(i)}))

    @property
    def x(self):
        return self[0]  # noqa: E704

    @property
    def y(self):
        return self[1]  # noqa: E704

    @property
    def z(self):
        return self[2]  # noqa: E704

    def norm(self):
        return Measure(OpNode(op="norm", inputs=[self.node]))  # noqa: E704

    # ---- time
    def rate(self):
        return Measure(OpNode(op="rate", inputs=[self.node]))  # noqa: E704

    def derivative(self):
        return Measure(OpNode(op="derivative", inputs=[self.node]))  # noqa: E704

    def integral(self):
        return Measure(OpNode(op="integral", inputs=[self.node]))  # noqa: E704

    def running_max(self):
        return Measure(OpNode(op="running_max", inputs=[self.node]))  # noqa: E704

    def running_min(self):
        return Measure(OpNode(op="running_min", inputs=[self.node]))  # noqa: E704

    def at_reset(self):
        return Measure(OpNode(op="at_reset", inputs=[self.node]))  # noqa: E704

    def since_reset(self):
        return Measure(OpNode(op="since_reset", inputs=[self.node]))  # noqa: E704

    def duration_true(self):
        return Measure(OpNode(op="duration_true", inputs=[self.node]))  # noqa: E704

    def time_since(self):
        return Measure(OpNode(op="time_since", inputs=[self.node]))  # noqa: E704

    def window(self, seconds: float) -> Window:
        return Window(self, float(seconds))


class Window:
    def __init__(self, m: Measure, seconds: float):
        self._m, self._s = m, seconds

    def _stat(self, stat: str) -> Measure:
        return Measure(
            OpNode(op="window", inputs=[self._m.node], args={"seconds": self._s, "stat": stat})
        )

    def mean(self):
        return self._stat("mean")  # noqa: E704

    def min(self):
        return self._stat("min")  # noqa: E704

    def max(self):
        return self._stat("max")  # noqa: E704

    def std(self):
        return self._stat("std")  # noqa: E704

    def sum(self):
        return self._stat("sum")  # noqa: E704


def as_measure(x) -> Measure:
    if isinstance(x, Measure):
        return x
    if isinstance(x, (bool, np.bool_)):
        return Measure(ConstNode(value=bool(x)))
    if isinstance(x, (int, float, np.integer, np.floating)):
        return Measure(ConstNode(value=float(x)))
    if isinstance(x, (list, tuple, np.ndarray)):
        return Measure(ConstNode(value=[float(v) for v in np.asarray(x, dtype=float).ravel()]))
    raise MeasureError(f"cannot use {type(x).__name__} as a measure (use lobs.props / lobs.rel)")
