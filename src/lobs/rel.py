"""Relative measures between 2 or 3 things. The whole vocabulary is *distance* and *angle*.

    pair = rel.between(a, b)
    pair.distance()
    pair.angle("polar", axis="z")       # also: elevation, azimuth, axes, rotation
    rel.between(a, v, b).angle()        # interior angle at v

A "thing" is an anchor, an actor or body (meaning its centre of mass), or ``lobs.WORLD``.
Degenerate geometry (coincident points, zero projection) yields NaN, never 0.
"""

from __future__ import annotations

import numpy as np

from .errors import MeasureError
from .measure.nodes import Measure
from .schema.measure import RelNode


def _path(x) -> str:
    return str(getattr(x, "path", x))


def _axis(a):
    return a if isinstance(a, str) else tuple(float(v) for v in np.asarray(a, dtype=float))


class Pair:
    def __init__(self, a, b):
        self.things = [_path(a), _path(b)]

    def distance(self) -> Measure:
        return Measure(RelNode(measure="distance", things=self.things))

    def angle(self, kind: str, axis="z", frame=None, axis_b=None) -> Measure:
        """kind: polar | elevation | azimuth | axes | rotation.

        polar/elevation/azimuth measure the direction a->b relative to ``axis`` of ``frame``
        ('a' (default), 'world', 'b' or any thing). 'axes' is the angle between a's ``axis`` and
        b's ``axis_b`` (default: same axis). 'rotation' is the geodesic angle between orientations.
        """
        return Measure(
            RelNode(
                measure="angle",
                things=self.things,
                angle_kind=kind,
                axis=_axis(axis),
                axis_b=None if axis_b is None else _axis(axis_b),
                frame=None if frame is None else _path(frame),
            )  # fmt: skip
        )


class Triple:
    def __init__(self, a, v, b):
        self.things = [_path(a), _path(v), _path(b)]

    def angle(self, signed_about=None, frame=None) -> Measure:
        """Interior angle at the middle thing; with ``signed_about`` a signed angle in (-pi, pi]."""
        return Measure(
            RelNode(
                measure="angle",
                things=self.things,
                signed_about=None if signed_about is None else _axis(signed_about),
                frame=None if frame is None else _path(frame),
            )  # fmt: skip
        )


def between(*things):
    if len(things) == 2:
        return Pair(*things)
    if len(things) == 3:
        return Triple(*things)
    raise MeasureError("rel.between needs 2 things (pair) or 3 things (a, vertex, b)")


# ---------------------------------------------------------------- recipes (distance + angle only)
class recipes:  # noqa: N801
    @staticmethod
    def along(a, b, axis="z", frame=None) -> Measure:
        """Signed distance of b from a measured along ``axis`` (point-to-plane distance)."""
        from . import ops

        p = between(a, b)
        return p.distance() * ops.cos(p.angle("polar", axis, frame))

    @staticmethod
    def off_axis(a, b, axis="z", frame=None) -> Measure:
        """Distance of b from the line through a along ``axis`` (point-to-line distance)."""
        from . import ops

        p = between(a, b)
        return p.distance() * ops.sin(p.angle("polar", axis, frame))

    @staticmethod
    def strain(a, b) -> Measure:
        d = between(a, b).distance()
        return d / d.at_reset() - 1.0

    @staticmethod
    def bend(a, v, b) -> Measure:
        return np.pi - between(a, v, b).angle()

    @staticmethod
    def curvature(a, v, b) -> Measure:
        """Menger curvature of three anchors: 2 sin(interior angle at v) / |a - b|."""
        from . import ops

        return 2.0 * ops.sin(between(a, v, b).angle()) / between(a, b).distance()
