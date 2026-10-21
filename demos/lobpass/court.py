"""Court geometry helpers (court frame: origin under the rim centre, +x toward half court)."""

from __future__ import annotations

import math

import numpy as np

from . import constants as C

RIM_XY = np.zeros(2)


def in_bounds(x: float, y: float, margin: float = 0.0) -> bool:
    return (C.BASELINE_X + margin <= x <= C.HALFCOURT_X - margin) and abs(
        y
    ) <= C.SIDELINE_Y - margin


def clamp_to_bounds(p, margin: float = 0.0):
    p = np.array(p, dtype=float)
    p[0] = min(max(p[0], C.BASELINE_X + margin), C.HALFCOURT_X - margin)
    p[1] = min(max(p[1], -C.SIDELINE_Y + margin), C.SIDELINE_Y - margin)
    return p


def polar(r: float, theta: float) -> np.ndarray:
    """Point at distance r from the rim, theta radians from +x (positive toward +y)."""
    return np.array([r * math.cos(theta), r * math.sin(theta)])


def _arc(r, a0, a1, n=48):
    t = np.linspace(a0, a1, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], axis=1)


def line_polylines() -> dict[str, list[np.ndarray]]:
    """Court markings as 2-D polylines (metres), consumed by the renderer. Keys name each marking."""
    yl, xb, xh, ys = C.LANE_HALF_WIDTH, C.BASELINE_X, C.HALFCOURT_X, C.SIDELINE_Y
    ft = C.FT_LINE_X
    three_a = math.asin(C.THREE_CORNER_Y / C.THREE_R)
    return {
        "boundary": [np.array([[xb, -ys], [xh, -ys], [xh, ys], [xb, ys], [xb, -ys]])],
        "lane": [np.array([[xb, -yl], [ft, -yl], [ft, yl], [xb, yl]])],
        "ft_circle": [_arc(C.FT_CIRCLE_R, -math.pi / 2, math.pi / 2) + np.array([ft, 0.0])],
        "three": [
            np.array([[xb, -C.THREE_CORNER_Y], [C.THREE_CORNER_X, -C.THREE_CORNER_Y]]),
            _arc(C.THREE_R, -three_a, three_a),
            np.array([[C.THREE_CORNER_X, C.THREE_CORNER_Y], [xb, C.THREE_CORNER_Y]]),
        ],
        "restricted": [_arc(C.RESTRICTED_R, -math.pi / 2, math.pi / 2)],
        "midcourt": [np.array([[xh, -ys], [xh, ys]])],
    }


def rim_ring_distance(px: float, py: float, pz: float) -> float:
    """Distance from a point to the rim tube centre-line (a circle of radius RIM_RING_R at z = RIM_TUBE_Z)."""
    d = math.hypot(px, py) - C.RIM_RING_R
    return math.hypot(d, pz - C.RIM_TUBE_Z)


def board_box_distance(p) -> float:
    """Distance from a point to the backboard box (0 inside)."""
    c = np.array(C.BOARD_CENTER)
    h = np.array(C.BOARD_SIZE) / 2
    q = np.maximum(np.abs(np.asarray(p, float) - c) - h, 0.0)
    return float(np.linalg.norm(q))
