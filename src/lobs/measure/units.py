"""Compile-time unit algebra (pint). Runtime values are plain float64 arrays."""

from __future__ import annotations

import pint

ureg = pint.UnitRegistry()

DIMENSIONLESS = ureg.Unit("dimensionless")
RADIAN = ureg.Unit("radian")
SECOND = ureg.Unit("second")


def parse(unit: str | None):
    return None if unit is None else ureg.Unit(unit)


def compatible(a, b) -> bool:
    return a.dimensionality == b.dimensionality


def fmt(unit) -> str:
    return "literal" if unit is None else str(unit)
