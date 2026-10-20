"""Measures dock (live values, add/edit by expression) and the live plot dock."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import exprs
from ..core.plotdata import format_value, traces

COLORS = ["#4aa3ff", "#ff6b6b", "#7bd88f", "#ffd166", "#c792ea", "#4dd0e1", "#ff9e64", "#a3be8c"]


class MeasuresPanel(QWidget):
    plot_changed = Signal()

    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.session, self.ctl = session, ctl
        self.plotted: set[str] = set()
        lay = QVBoxLayout(self)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Plot", "Name", "Value", "Unit", "Expression"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.itemChanged.connect(self._item_changed)
        self.table.itemSelectionChanged.connect(self._fill_editor)
        lay.addWidget(self.table)
        row = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("name")
        self.name_edit.setMaximumWidth(120)
        self.expr_edit = QLineEdit()
        self.expr_edit.setPlaceholderText("props.height(ball) < 0.5   |   rel.between(a, b).distance()")
        self.expr_edit.textChanged.connect(self._check)
        self.expr_edit.returnPressed.connect(self.add)
        self.add_btn = QPushButton("Add / update")
        self.add_btn.clicked.connect(self.add)
        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self.delete)
        for w in (self.name_edit, self.expr_edit, self.add_btn, self.del_btn):
            row.addWidget(w)
        lay.addLayout(row)
        self.msg = QLabel("")
        lay.addWidget(self.msg)
        session.subscribe(lambda _k: self.rebuild())
        ctl.subscribe(self._on_sim)
        self.rebuild()

    # ------------------------------------------------------------ table
    def rebuild(self) -> None:
        spec = self.session.world.spec
        names = [n for n in spec.measures if not n.startswith("__")]
        self.table.blockSignals(True)
        self.table.setRowCount(len(names))
        meta = self.ctl.sim.evaluator.meta() if self.ctl.sim is not None else {}
        for r, n in enumerate(names):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if n in self.plotted else Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, chk)
            for c, text in ((1, n), (2, ""), (3, meta.get(n, {}).get("unit", "")), (4, exprs.source_of(spec, n) or "(built in code)")):
                it = QTableWidgetItem(text)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.table.setItem(r, c, it)
        self.table.blockSignals(False)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.plotted &= set(names)

    def _on_sim(self, kind: str) -> None:
        if kind == "built":
            self.rebuild()
        elif kind in ("stepped", "reset"):
            self.update_values()

    def update_values(self) -> None:
        sim = self.ctl.sim
        if sim is None:
            return
        vals = sim.measurements()
        for r in range(self.table.rowCount()):
            n = self.table.item(r, 1).text()
            if n in vals:
                self.table.item(r, 2).setText(format_value(vals[n]))

    def _item_changed(self, item) -> None:
        if item.column() != 0:
            return
        name = self.table.item(item.row(), 1).text()
        (self.plotted.add if item.checkState() == Qt.CheckState.Checked else self.plotted.discard)(name)
        self.plot_changed.emit()

    # ------------------------------------------------------------ editing
    def _fill_editor(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            n = self.table.item(rows[0].row(), 1).text()
            self.name_edit.setText(n)
            self.expr_edit.setText(exprs.source_of(self.session.world.spec, n) or "")

    def _check(self, text: str) -> bool:
        if not text.strip():
            self.msg.setText("")
            return False
        try:
            exprs.evaluate(text, self.session.world)
        except exprs.ExprError as e:
            self.msg.setStyleSheet("color: #e57373")
            self.msg.setText(str(e))
            return False
        self.msg.setStyleSheet("color: #81c784")
        self.msg.setText("ok")
        return True

    def add(self) -> None:
        name, text = self.name_edit.text().strip(), self.expr_edit.text()
        if not name or not name.replace("_", "").replace(".", "").isalnum():
            self.msg.setStyleSheet("color: #e57373")
            self.msg.setText("give the measure a name (letters, digits, _ and .)")
            return
        if not self._check(text):
            return
        self.session.edit(f"measure {name}", lambda sp: exprs.set_measure(sp, name, text, self.session.world))

    def delete(self) -> None:
        name = self.name_edit.text().strip()
        if name in self.session.world.spec.measures:
            self.session.edit(f"delete measure {name}", lambda sp: exprs.remove_measure(sp, name))


class PlotPanel(QWidget):
    """Live traces of the plotted measures over a sliding window."""

    def __init__(self, ctl, measures: MeasuresPanel, parent=None):
        super().__init__(parent)
        self.ctl, self.measures = ctl, measures
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.window_box = QLineEdit("10")
        self.window_box.setMaximumWidth(50)
        self.follow = QCheckBox("last")
        self.follow.setChecked(True)
        bar.addWidget(self.follow)
        bar.addWidget(self.window_box)
        bar.addWidget(QLabel("s"))
        bar.addStretch()
        self.csv_btn = QPushButton("Export CSV")
        bar.addWidget(self.csv_btn)
        lay.addLayout(bar)
        pg.setConfigOptions(antialias=True, background=(27, 29, 33), foreground=(220, 220, 220))
        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(10, 10))
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "time", "s")
        lay.addWidget(self.plot)
        self.curves: dict[str, pg.PlotDataItem] = {}
        ctl.subscribe(lambda k: self.refresh() if k in ("reset", "built") else None)
        measures.plot_changed.connect(self.refresh)

    def window(self) -> float | None:
        if not self.follow.isChecked():
            return None
        try:
            return max(float(self.window_box.text()), 0.1)
        except ValueError:
            return 10.0

    def refresh(self) -> None:
        sim = self.ctl.sim
        wanted: dict[str, tuple] = {}
        if sim is not None:
            for name in sorted(self.measures.plotted):
                if name in sim.history:
                    for lab, t, y in traces(sim.history, name, self.window()):
                        wanted[lab] = (t, y)
        for lab in [k for k in self.curves if k not in wanted]:
            self.plot.removeItem(self.curves.pop(lab))
        for i, (lab, (t, y)) in enumerate(wanted.items()):
            if lab not in self.curves:
                pen = pg.mkPen(QColor(COLORS[len(self.curves) % len(COLORS)]), width=2)
                self.curves[lab] = self.plot.plot(name=lab, pen=pen)
            self.curves[lab].setData(t, y)
