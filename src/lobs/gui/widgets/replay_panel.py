"""Replay panel: scrub a recorded run, add measures after the fact and plot them."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..core.replay import Replay
from .timeline import Timeline

SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]


class ReplayPanel(QWidget):
    def __init__(self, ctl, viewport, parent=None):
        super().__init__(parent)
        self.ctl, self.viewport = ctl, viewport
        self.replay: Replay | None = None
        self.playing = False
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.open_btn = QPushButton("Open recording...")
        self.open_btn.clicked.connect(self.open_dialog)
        self.last_btn = QPushButton("Replay last run")
        self.last_btn.clicked.connect(self.from_live)
        self.save_btn = QPushButton("Save run...")
        self.save_btn.clicked.connect(self.save_dialog)
        self.live_btn = QPushButton("Back to live")
        self.live_btn.clicked.connect(self.stop_replay)
        for b in (self.open_btn, self.last_btn, self.save_btn, self.live_btn):
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        tr = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._slider)
        self.time_label = QLabel("")
        self.speed = QComboBox()
        for s in SPEEDS:
            self.speed.addItem(f"{s:g}x", s)
        self.speed.setCurrentIndex(SPEEDS.index(1.0))
        self.loop = QCheckBox("loop")
        self.loop.setChecked(True)
        self.loop.toggled.connect(self._loop_changed)
        for w in (self.play_btn, self.slider, self.time_label, self.speed, self.loop):
            tr.addWidget(w, 1 if w is self.slider else 0)
        lay.addLayout(tr)
        self.timeline = Timeline()
        self.timeline.seek.connect(self.seek_time)
        lay.addWidget(self.timeline)
        ev = QHBoxLayout()
        self.expr = QLineEdit()
        self.expr.setPlaceholderText(
            "evaluate a measure on this recording, e.g. rel.between(a, b).distance()"
        )
        self.expr.returnPressed.connect(self.evaluate)
        self.go = QPushButton("Evaluate")
        self.go.clicked.connect(self.evaluate)
        ev.addWidget(self.expr, 1)
        ev.addWidget(self.go)
        lay.addLayout(ev)
        self.msg = QLabel("")
        lay.addWidget(self.msg)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen("#4aa3ff", width=2))
        self.plot.addItem(self.cursor)
        lay.addWidget(self.plot, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _loop_changed(self, v: bool) -> None:
        if self.replay is not None:
            self.replay.loop = v

    # ------------------------------------------------------------ sources
    def open_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open recording", "", "Recording (*.npz)")
        if path:
            self.load(path)

    def load(self, path) -> None:
        try:
            self.start(Replay.load(path))
        except Exception as e:  # noqa: BLE001
            self._error(str(e))

    def from_live(self) -> None:
        sim = self.ctl.sim
        if sim is None or len(sim.history) < 2:
            self._error("run the simulation first (nothing recorded yet)")
            return
        self.ctl.pause()
        self.start(Replay.from_sim(sim))

    def save_dialog(self) -> None:
        sim = self.ctl.sim
        if sim is None or len(sim.history) < 2:
            self._error("nothing recorded yet")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save recording", "", "Recording (*.npz)")
        if path:
            sim.export(path)

    def start(self, replay: Replay) -> None:
        self.replay = replay
        replay.loop = self.loop.isChecked()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(len(replay) - 1, 0))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.timeline.set_lanes(replay.lanes(), replay.duration)
        self.viewport.show_replay(replay)
        self._sync_ui()
        self.msg.setStyleSheet("")
        self.msg.setText(
            f"{len(replay)} frames, {replay.duration:.2f} s, recorded: {', '.join(replay.recorded_measures()[:6])}"
        )

    def stop_replay(self) -> None:
        self.playing = False
        self.replay = None
        self.play_btn.setText("Play")
        self.viewport.clear_replay()

    # ------------------------------------------------------------ transport
    def toggle(self) -> None:
        if self.replay is None:
            return
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def _tick(self) -> None:
        if self.replay is None or not self.playing:
            return
        if not self.replay.advance(0.033 * self.speed.currentData()):
            self.playing = False
            self.play_btn.setText("Play")
        self._sync_ui()
        self.viewport.refresh()

    def _slider(self, i: int) -> None:
        if self.replay is not None:
            self.replay.index = int(i)
            self._sync_ui(from_slider=True)
            self.viewport.refresh()

    def seek_time(self, t: float) -> None:
        if self.replay is not None:
            self.replay.seek_time(t)
            self._sync_ui()
            self.viewport.refresh()

    def _sync_ui(self, from_slider: bool = False) -> None:
        if self.replay is None:
            return
        if not from_slider:
            self.slider.blockSignals(True)
            self.slider.setValue(self.replay.index)
            self.slider.blockSignals(False)
        t = float(self.replay.times[self.replay.index])
        self.time_label.setText(f"{t:6.3f} / {self.replay.duration:.3f} s")
        self.timeline.set_playhead(t)
        self.cursor.setValue(t)

    # ------------------------------------------------------------ evaluate
    def evaluate(self) -> None:
        if self.replay is None:
            self._error("open or start a replay first")
            return
        try:
            y = self.replay.evaluate(self.expr.text())
        except Exception as e:  # noqa: BLE001 - library errors (live-only measures, contacts) are shown as is
            self._error(str(e))
            return
        self.msg.setStyleSheet("color: #81c784")
        if y.ndim == 1:
            self.msg.setText(
                f"{len(y)} samples   min {y.min():.5g}   max {y.max():.5g}   final {y[-1]:.5g}"
            )
        else:
            self.msg.setText(f"{len(y)} samples of shape {y.shape[1:]}")
        self.plot.clear()
        self.plot.addItem(self.cursor)
        flat = y.reshape(len(y), -1).astype(float)
        for k in range(flat.shape[1]):
            self.plot.plot(self.replay.times, flat[:, k], pen=pg.mkPen(k, width=2))

    def _error(self, text: str) -> None:
        self.msg.setStyleSheet("color: #e57373")
        self.msg.setText(text)
