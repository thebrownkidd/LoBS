"""Force laws and adversaries: live budget bars, runtime switches and a one-click budgeted gust."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.gust import inject_gust

COLS = ["On", "Name", "Target", "Kind", "Impulse", "Peak N", "Active s", "Now"]


class ForcesPanel(QWidget):
    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.session, self.ctl = session, ctl
        self._n = 0
        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        self.actor = QComboBox()
        self.dir = [QDoubleSpinBox() for _ in range(3)]
        for w, v in zip(self.dir, (1.0, 0.0, 0.0), strict=True):
            w.setRange(-1e3, 1e3)
            w.setDecimals(2)
            w.setValue(v)
        self.peak = QDoubleSpinBox()
        self.peak.setRange(0.01, 1e6)
        self.peak.setValue(20.0)
        self.impulse = QDoubleSpinBox()
        self.impulse.setRange(0.001, 1e6)
        self.impulse.setValue(2.0)
        self.btn = QPushButton("Inject gust")
        self.btn.clicked.connect(self.inject)
        for w in (
            QLabel("Gust on"),
            self.actor,
            QLabel("dir"),
            *self.dir,
            QLabel("peak N"),
            self.peak,
            QLabel("impulse N s"),
            self.impulse,
            self.btn,
        ):
            row.addWidget(w)
        row.addStretch()
        lay.addLayout(row)
        self.msg = QLabel("")
        lay.addWidget(self.msg)
        session.subscribe(lambda _k: self.rebuild())
        ctl.subscribe(self._on_sim)
        self.rebuild()

    def rebuild(self) -> None:
        spec = self.session.world.spec
        self.actor.clear()
        self.actor.addItems([a.name for a in spec.actors])
        self.table.setRowCount(len(spec.forces))
        for r, f in enumerate(spec.forces):
            chk = QCheckBox()
            chk.setChecked(True)
            chk.toggled.connect(lambda on, name=f.name: self._switch(name, on))
            self.table.setCellWidget(r, 0, chk)
            budget = f.budget
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setFormat("no limit" if budget is None or budget.impulse is None else "%p%")
            self.table.setCellWidget(r, 4, bar)
            vals = [
                f.name,
                f.on,
                "adversary" if f.adversary else "force",
                None,
                "" if budget is None or budget.peak is None else f"{budget.peak:g}",
                "",
                "",
            ]
            for c, v in zip((1, 2, 3, 4, 5, 6, 7), vals, strict=True):
                if v is not None:
                    it = QTableWidgetItem(v)
                    it.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()

    def _switch(self, name: str, on: bool) -> None:
        sim = self.ctl.sim
        if sim is not None:
            for law in sim._laws:
                if law.name == name:
                    law.enabled = on

    def _on_sim(self, kind: str) -> None:
        if kind == "stepped":
            self._n += 1
            if self._n % 5:
                return
        if kind in ("stepped", "reset", "built"):
            self.update_live()

    def update_live(self) -> None:
        sim = self.ctl.sim
        if sim is None:
            return
        stats = sim.force_stats
        for r, f in enumerate(self.session.world.spec.forces):
            st = stats.get(f.name)
            if st is None:
                continue
            bar = self.table.cellWidget(r, 4)
            if f.budget is not None and f.budget.impulse:
                bar.setValue(int(1000 * min(st["impulse"] / f.budget.impulse, 1.0)))
                bar.setFormat(f"{st['impulse']:.3g} / {f.budget.impulse:g} N s")
            else:
                bar.setFormat(f"{st['impulse']:.3g} N s")
            self.table.item(r, 5).setText(
                f"{st['peak']:.3g}"
                + (f" / {f.budget.peak:g}" if f.budget and f.budget.peak else "")
            )
            limit = f"  / {f.budget.active_time:g}" if f.budget and f.budget.active_time else ""
            self.table.item(r, 6).setText(f"{st['active_time']:.2f}{limit}")
            self.table.item(r, 7).setText(
                f"{np.linalg.norm(st['last_force']):.3g} N" if st["active"] else "off"
            )

    def inject(self) -> None:
        sim = self.ctl.sim
        actor = self.actor.currentText()
        if sim is None or not actor:
            self.msg.setText("build the simulation first and pick an actor")
            return
        try:
            info = inject_gust(
                sim, actor, [w.value() for w in self.dir], self.peak.value(), self.impulse.value()
            )
            self.msg.setStyleSheet("color: #81c784")
            self.msg.setText(
                f"gust: {info['force']:.3g} N for {info['seconds']:.3f} s = {info['impulse']:.3g} N s"
            )
        except Exception as e:  # noqa: BLE001
            self.msg.setStyleSheet("color: #e57373")
            self.msg.setText(str(e))
