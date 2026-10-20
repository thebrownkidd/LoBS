"""Timeline: behavior-state and force lanes over time, with a playhead; click or drag to seek."""

from __future__ import annotations

import colorsys

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..core.replay import Lane, segments

LANE_H, LEFT, TOP = 26, 130, 18


def state_color(i: int) -> QColor:
    r, g, b = colorsys.hsv_to_rgb((i * 0.381966) % 1.0, 0.45, 0.62)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


class Timeline(QWidget):
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lanes: list[Lane] = []
        self.playhead = 0.0
        self.t_end = 1.0
        self.setMinimumHeight(70)

    def set_lanes(self, lanes: list[Lane], t_end: float | None = None) -> None:
        self.lanes = lanes
        ends = [float(ln.t[-1]) for ln in lanes if len(ln.t)]
        self.t_end = max(t_end if t_end is not None else (max(ends) if ends else 1.0), 1e-6)
        self.setMinimumHeight(TOP + LANE_H * max(len(lanes), 1) + 8)
        self.update()

    def set_playhead(self, t: float) -> None:
        self.playhead = t
        self.update()

    def _x(self, t: float) -> float:
        return LEFT + (self.width() - LEFT - 8) * t / self.t_end

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(27, 29, 33))
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        if not self.lanes:
            p.setPen(QColor(150, 150, 150))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "no behaviors or force laws to show"
            )
            return
        p.setPen(QColor(120, 125, 135))
        for k in range(6):  # time ticks
            t = self.t_end * k / 5
            p.drawText(int(self._x(t)) - 12, 12, f"{t:.2g}s")
        for r, lane in enumerate(self.lanes):
            y = TOP + r * LANE_H
            p.setPen(QColor(210, 214, 220))
            p.drawText(6, y + 17, lane.label[:20])
            p.fillRect(QRectF(LEFT, y + 3, self.width() - LEFT - 8, LANE_H - 6), QColor(36, 39, 45))
            if lane.kind == "state":
                for t0, t1, s in segments(lane):
                    rect = QRectF(
                        self._x(t0), y + 3, max(self._x(t1) - self._x(t0), 1.0), LANE_H - 6
                    )
                    p.fillRect(rect, state_color(s))
                    if rect.width() > 36 and s < len(lane.names):
                        p.setPen(QColor(255, 255, 255))
                        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, lane.names[s])
            elif len(lane.t):
                peak = float(np.max(lane.v)) or 1.0
                step = max(len(lane.t) // max(self.width(), 1), 1)
                p.setPen(QPen(QColor(235, 90, 90), 1))
                for i in range(0, len(lane.t), step):
                    h = (LANE_H - 8) * float(lane.v[i]) / peak
                    if h > 0.4:
                        x = self._x(float(lane.t[i]))
                        p.drawLine(int(x), int(y + LANE_H - 4), int(x), int(y + LANE_H - 4 - h))
        x = self._x(self.playhead)
        p.setPen(QPen(QColor(90, 200, 255), 2))
        p.drawLine(int(x), TOP - 4, int(x), self.height() - 2)

    def _seek_at(self, x: float) -> None:
        t = float(np.clip((x - LEFT) / max(self.width() - LEFT - 8, 1) * self.t_end, 0, self.t_end))
        self.seek.emit(t)

    def mousePressEvent(self, e):
        self._seek_at(e.position().x())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._seek_at(e.position().x())
