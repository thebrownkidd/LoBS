"""Anthropometrics: reach, jump physics and dunk feasibility."""

from __future__ import annotations

import math

from . import constants as C


def reach(h: float) -> float:
    return C.REACH_FRAC * h


def jump_vz(lift: float) -> float:
    return math.sqrt(2.0 * C.G * lift)


def jump_cmd_vz(lift: float) -> float:
    """Take-off speed to command with a one-tick impulse so the apex is exactly ``lift``: the body also climbs
    vz * TICK / 2 while the impulse ramps up, so solve vz^2 / 2g + vz * TICK / 2 = lift."""
    a = C.G * C.TICK / 2.0
    return -a + math.sqrt(a * a + 2.0 * C.G * lift)


def apex_time(lift: float) -> float:
    return jump_vz(lift) / C.G


def min_dunk_lift(h: float) -> float:
    """Smallest vertical leap letting a player of height h reach C.DUNK_REACH_Z."""
    return max(0.0, C.DUNK_REACH_Z - reach(h))


def hand_height_at_apex(h: float, lift: float) -> float:
    return reach(h) + lift
