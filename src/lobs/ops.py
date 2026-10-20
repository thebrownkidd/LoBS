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


# ---- triggers and randomness (see docs/behaviors.md)
def latch(set_, reset) -> Measure:
    """SR latch: becomes True when ``set_`` is true, False when ``reset`` is true (reset wins)."""
    return _op("latch", set_, reset)


def chance(p: float) -> Measure:
    """True with probability ``p`` on each tick, from the run RNG (use ``.rising()`` to gate per event)."""
    return _op("chance", p=float(p))


def uniform(lo: float, hi: float, unit: str | None = None) -> Measure:
    """A uniform random number resampled every tick (pair with ``.sample_on(trigger)`` to sample once)."""
    return _op("uniform", lo=float(lo), hi=float(hi), unit=unit)


def normal(mu: float, sd: float, unit: str | None = None) -> Measure:
    return _op("normal", mu=float(mu), sd=float(sd), unit=unit)


# ---- vector helpers
def vec(x, y, z) -> Measure:
    """Build a 3-vector measure from three scalar measures / numbers."""
    return _op("vec", x, y, z)


def unit(v) -> Measure:
    """Unit vector of a 3-vector measure; the zero vector (never NaN) when its length is ~0."""
    return _op("unit", v)


def clip_norm(v, limit) -> Measure:
    """Scale a 3-vector measure down so its length is at most ``limit`` (a number or scalar measure)."""
    return _op("clip_norm", v, limit)


def event(name: str) -> Measure:
    """True for the one tick after a behavior ``Emit`` of ``name`` (live runs only)."""
    return _op("event", name=name)


def as_unit(m, unit: str) -> Measure:
    """Reinterpret a number measure as ``unit`` (same values). Use it to give gains their units, e.g.
    ``ops.as_unit(k * (target - pos), "newton")`` for a spring law."""
    return _op("as_unit", m, unit=unit)
