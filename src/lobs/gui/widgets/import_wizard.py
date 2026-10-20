"""Import wizard: choose a foreign file, choose what moves, review every physics decision, override, open."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ..core import importing as imp
from ..core.jobs import Job

COLS = ["Target", "Field", "Value", "Provenance", "Confidence", "Why", "Review"]
CONF_COLOR = {"low": QColor(230, 110, 110), "medium": QColor(225, 180, 80)}


class ImportWizard(QDialog):
    imported = Signal(object, str)  # World, package path

    def __init__(self, parent=None, source: str | None = None):
        super().__init__(parent)
        self.setWindowTitle("Import a world")
        self.resize(1000, 640)
        self.req = imp.ImportRequest(source or "")
        self.world = None
        self.report = None
        self.job = None
        lay = QVBoxLayout(self)
        form = QFormLayout()
        row = QHBoxLayout()
        self.path_edit = QLineEdit(source or "")
        self.path_edit.textChanged.connect(self._update)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self.browse)
        row.addWidget(self.path_edit)
        row.addWidget(browse)
        form.addRow("Source file", row)
        self.motion = QComboBox()
        self.motion.addItem("(choose what moves)", None)
        self.motion.addItem("auto: guess from class, name and size", "auto")
        self.motion.addItem("dynamic: everything movable except ground support", "dynamic")
        self.motion.currentIndexChanged.connect(self._update)
        form.addRow("Motion mode (required)", self.motion)
        self.profile = QComboBox()
        self.profile.addItems(["generic", "building", "robotics"])
        self.units = QComboBox()
        self.units.addItems(["auto", "m", "mm", "cm", "ft", "in"])
        self.up = QComboBox()
        self.up.addItems(["auto", "z", "y"])
        self.scale = QDoubleSpinBox()
        self.scale.setRange(1e-6, 1e6)
        self.scale.setDecimals(6)
        self.scale.setValue(1.0)
        opts = QHBoxLayout()
        for label, w in (("profile", self.profile), ("units", self.units), ("up axis", self.up), ("scale", self.scale)):
            opts.addWidget(QLabel(label))
            opts.addWidget(w)
        opts.addStretch()
        form.addRow("Options", opts)
        lay.addLayout(form)
        run = QHBoxLayout()
        self.run_btn = QPushButton("Import")
        self.run_btn.clicked.connect(self.start)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setTextVisible(False)
        self.summary = QLabel("")
        run.addWidget(self.run_btn)
        run.addWidget(self.progress, 1)
        lay.addLayout(run)
        lay.addWidget(self.summary)
        flt = QHBoxLayout()
        self.only_review = QCheckBox("only decisions that need review")
        self.only_review.setChecked(True)
        self.only_review.toggled.connect(self.fill_table)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("filter...")
        self.filter.textChanged.connect(self.fill_table)
        flt.addWidget(self.only_review)
        flt.addWidget(self.filter, 1)
        lay.addLayout(flt)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.itemChanged.connect(self._edited)
        lay.addWidget(self.table, 1)
        self.msg = QLabel("Edit the Value of a motion / mass / density / friction / restitution / collision decision, then re-import.")
        lay.addWidget(self.msg)
        bottom = QHBoxLayout()
        self.reimport_btn = QPushButton("Re-import with overrides")
        self.reimport_btn.clicked.connect(self.start)
        self.open_btn = QPushButton("Open in Studio")
        self.open_btn.clicked.connect(self.accept_open)
        cancel = QPushButton("Close")
        cancel.clicked.connect(self.reject)
        for b in (self.reimport_btn, self.open_btn, cancel):
            bottom.addWidget(b)
        lay.addLayout(bottom)
        self._filling = False
        self._update()

    # ------------------------------------------------------------ inputs
    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a world file", "", imp.file_filter())
        if path:
            self.path_edit.setText(path)

    def _update(self, *_):
        self.req.source = self.path_edit.text().strip()
        self.req.motion = self.motion.currentData()
        self.run_btn.setEnabled(self.req.ready() and self.job is None)
        self.reimport_btn.setEnabled(self.report is not None and self.req.ready() and self.job is None)
        self.open_btn.setEnabled(self.world is not None)

    def _collect(self) -> None:
        self.req.profile = self.profile.currentText()
        self.req.units = self.units.currentText()
        self.req.up_axis = self.up.currentText()
        self.req.scale = self.scale.value()
        if not self.req.out and self.req.source:
            self.req.out = str(Path(self.req.source).with_suffix(".lobs"))

    # ------------------------------------------------------------ running
    def start(self, sync: bool = False) -> None:
        self._update()
        if not self.req.ready():
            return
        self._collect()
        self.progress.setRange(0, 0)
        self.msg.setText("importing...")
        if sync is True:
            try:
                self._done(imp.run_import(self.req))
            except Exception as e:  # noqa: BLE001
                self._failed(f"{type(e).__name__}: {e}")
            return
        self.job = Job(imp.run_import, self.req, parent=self)
        self.job.succeeded.connect(self._done)
        self.job.failed.connect(self._failed)
        self.run_btn.setEnabled(False)
        self.job.start()

    def _done(self, result) -> None:
        self.world, self.report = result
        self.job = None
        self.progress.setRange(0, 1)
        c = self.report.counts
        review = len(self.report.needs_review)
        self.summary.setText(
            f"{self.report.format}: " + ", ".join(f"{k}={v}" for k, v in c.items()) + f"   |   {review} decisions need review, {len(self.report.dropped)} features not imported"
        )
        self.msg.setText("Imported. Review, override and re-import, or open it.")
        self.fill_table()
        self._update()

    def _failed(self, text: str) -> None:
        self.job = None
        self.progress.setRange(0, 1)
        self.msg.setStyleSheet("color: #e57373")
        self.msg.setText(text)
        self._update()

    # ------------------------------------------------------------ decision table
    def fill_table(self, *_):
        if self.report is None:
            return
        rows = imp.decision_rows(self.report, self.only_review.isChecked(), self.filter.text())
        self._filling = True
        self.table.setRowCount(len(rows))
        for r, d in enumerate(rows):
            cells = [d["target"], d["field"], str(d["value"]), d["provenance"], d["confidence"], d["rationale"], "review" if d["review"] else ""]
            for c, text in enumerate(cells):
                it = QTableWidgetItem(text)
                editable = c == 2 and d["editable"]
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | (Qt.ItemFlag.ItemIsEditable if editable else Qt.ItemFlag.NoItemFlags))
                if d["confidence"] in CONF_COLOR:
                    it.setForeground(CONF_COLOR[d["confidence"]])
                if self.req.overrides.get(d["target"], {}).get(d["field"].split(".")[0]) is not None and c == 2:
                    it.setText(str(self.req.overrides[d["target"]][d["field"].split(".")[0]]))
                    it.setBackground(QColor(40, 90, 60))
                self.table.setItem(r, c, it)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)
        self._filling = False

    def _edited(self, item) -> None:
        if self._filling or item.column() != 2:
            return
        target = self.table.item(item.row(), 0).text()
        field = self.table.item(item.row(), 1).text()
        try:
            imp.set_override(self.req.overrides, target, field, imp.parse_value(field, item.text()))
            self.msg.setStyleSheet("color: #81c784")
            self.msg.setText(f"override staged for {target}.{field.split('.')[0]}; press Re-import")
        except ValueError as e:
            self.msg.setStyleSheet("color: #e57373")
            self.msg.setText(str(e))

    def accept_open(self) -> None:
        if self.world is not None:
            self.imported.emit(self.world, self.req.out or "")
            self.accept()
