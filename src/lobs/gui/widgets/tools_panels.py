"""Console, world check, engine comparison and Monte Carlo panels."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core import tools
from ..core.jobs import Job


class _Runner:
    """Mixin: run a function on a worker thread, or inline when ``sync`` (tests, scripts)."""

    sync = False

    def run_job(self, fn, done, failed) -> None:
        if self.sync:
            try:
                done(fn())
            except Exception as e:  # noqa: BLE001
                failed(f"{type(e).__name__}: {e}")
            return
        job = Job(fn, parent=self)
        job.succeeded.connect(done)
        job.failed.connect(failed)
        job.finished.connect(lambda: setattr(self, "_job", None))
        self._job = job
        job.start()


class ConsolePanel(QWidget):
    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.console = tools.Console(lambda: session.world, lambda: ctl.sim, lambda: session)
        lay = QVBoxLayout(self)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setPlainText(
            "LoBS console: world, sim, session, props, rel, ops, act, np, lobs are in scope."
        )
        self.inp = QLineEdit()
        self.inp.setPlaceholderText(">>> ")
        self.inp.returnPressed.connect(self.submit)
        lay.addWidget(self.out)
        lay.addWidget(self.inp)
        self.history: list[str] = []
        self._pos = 0

    def submit(self) -> None:
        line = self.inp.text()
        if not line.strip():
            return
        self.history.append(line)
        self._pos = len(self.history)
        self.out.appendPlainText(f">>> {line}")
        res = self.console.run(line).rstrip()
        if res:
            self.out.appendPlainText(res)
        self.inp.clear()

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down) and self.history:
            self._pos = max(
                0, min(len(self.history), self._pos + (-1 if e.key() == Qt.Key.Key_Up else 1))
            )
            self.inp.setText(self.history[self._pos] if self._pos < len(self.history) else "")
        else:
            super().keyPressEvent(e)


class CheckPanel(QWidget, _Runner):
    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.backend = QComboBox()
        self.backend.addItems(ctl.available_backends())
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.1, 600)
        self.seconds.setValue(2.0)
        self.btn = QPushButton("Run stability check")
        self.btn.clicked.connect(self.start)
        for w in (QLabel("backend"), self.backend, QLabel("seconds"), self.seconds, self.btn):
            bar.addWidget(w)
        bar.addStretch()
        lay.addLayout(bar)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        lay.addWidget(self.out)

    def start(self) -> None:
        self.btn.setEnabled(False)
        self.out.setPlainText("running...")
        world, be, sec = self.session.world.copy(), self.backend.currentText(), self.seconds.value()
        self.run_job(lambda: tools.run_check(world, be, sec), self._done, self._failed)

    def _done(self, res) -> None:
        self.btn.setEnabled(True)
        lines = [res.summary()]
        for pair, depth in res.initial_penetrations[:10]:
            lines.append(f"  initial penetration {pair}: {depth * 1000:.1f} mm")
        self.out.setPlainText("\n".join(lines))

    def _failed(self, text: str) -> None:
        self.btn.setEnabled(True)
        self.out.setPlainText(text)


class ComparePanel(QWidget, _Runner):
    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.session = session
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.checks = []
        for b in ctl.available_backends():
            c = QCheckBox(b)
            c.setChecked(True)
            self.checks.append(c)
            bar.addWidget(c)
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.1, 600)
        self.seconds.setValue(2.0)
        self.btn = QPushButton("Compare engines")
        self.btn.clicked.connect(self.start)
        bar.addWidget(QLabel("seconds"))
        bar.addWidget(self.seconds)
        bar.addWidget(self.btn)
        bar.addStretch()
        lay.addLayout(bar)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        lay.addWidget(self.out)

    def start(self) -> None:
        names = [c.text() for c in self.checks if c.isChecked()]
        if len(names) < 2:
            self.out.setPlainText("select at least two backends")
            return
        self.btn.setEnabled(False)
        self.out.setPlainText("running...")
        world, sec = self.session.world.copy(), self.seconds.value()
        self.run_job(lambda: tools.run_compare(world, names, sec), self._done, self._failed)

    def _done(self, comp) -> None:
        self.btn.setEnabled(True)
        self.out.setPlainText(comp.table())

    def _failed(self, text: str) -> None:
        self.btn.setEnabled(True)
        self.out.setPlainText(text)


class MonteCarloPanel(QWidget, _Runner):
    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.session, self.ctl = session, ctl
        self.result = None
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self.measure = QComboBox()
        self.stat = QComboBox()
        self.stat.addItems(list(tools.STATS))
        self.runs = QSpinBox()
        self.runs.setRange(2, 5000)
        self.runs.setValue(20)
        self.seconds = QDoubleSpinBox()
        self.seconds.setRange(0.1, 600)
        self.seconds.setValue(2.0)
        self.btn = QPushButton("Run")
        self.btn.clicked.connect(self.start)
        for w in (
            QLabel("measure"),
            self.measure,
            QLabel("stat"),
            self.stat,
            QLabel("runs"),
            self.runs,
            QLabel("seconds"),
            self.seconds,
            self.btn,
        ):
            bar.addWidget(w)
        bar.addStretch()
        lay.addLayout(bar)
        self.random = QLineEdit()
        self.random.setPlaceholderText(
            'randomize, e.g. {"ball": {"pos": (lobs.Uniform(-1, 1), 0.0, 1.0)}}'
        )
        lay.addWidget(self.random)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.plot = pg.PlotWidget()
        self.plot.showGrid(y=True, alpha=0.25)
        lay.addWidget(self.plot, 1)
        self.bins = QListWidget()
        self.bins.setMaximumHeight(90)
        self.bins.itemClicked.connect(self._bin_clicked)
        lay.addWidget(self.bins)
        row = QHBoxLayout()
        self.replay_btn = QPushButton("Replay selected seed")
        self.replay_btn.clicked.connect(self.replay_seed)
        self.msg = QLabel("")
        row.addWidget(self.replay_btn)
        row.addWidget(self.msg, 1)
        lay.addLayout(row)
        session.subscribe(lambda _k: self.refresh_measures())
        self.refresh_measures()

    def refresh_measures(self) -> None:
        cur = self.measure.currentText()
        self.measure.clear()
        self.measure.addItems(
            [n for n in self.session.world.spec.measures if not n.startswith("__")]
        )
        if cur:
            self.measure.setCurrentText(cur)

    def _randomize(self):
        text = self.random.text().strip()
        if not text:
            return None
        import lobs

        ns = {"lobs": lobs, "Uniform": lobs.Uniform, "Normal": lobs.Normal, "Choice": lobs.Choice}
        return eval(text, {"__builtins__": {}}, ns)  # noqa: S307 - the user's own expression

    def start(self) -> None:
        if not self.measure.currentText():
            self.msg.setText("add a measure to the world first")
            return
        try:
            rnd = self._randomize()
        except Exception as e:  # noqa: BLE001
            self.msg.setStyleSheet("color: #e57373")
            self.msg.setText(f"randomize: {e}")
            return
        self.btn.setEnabled(False)
        self.progress.setRange(0, 0)
        world, be = self.session.world.copy(), self.ctl.backend
        args = dict(
            measure=self.measure.currentText(),
            runs=self.runs.value(),
            seconds=self.seconds.value(),
            backend=be,
            randomize=rnd,
            stat=self.stat.currentText(),
        )
        self.run_job(lambda: tools.monte_carlo(world, **args), self._done, self._failed)

    def _done(self, res: dict) -> None:
        self.result = res
        self.btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        counts, edges = res["histogram"]
        self.plot.clear()
        self.plot.addItem(
            pg.BarGraphItem(
                x=(edges[:-1] + edges[1:]) / 2,
                height=counts,
                width=float(np.diff(edges).mean()) * 0.9,
                brush="#4aa3ff",
            )
        )
        self.bins.clear()
        for i, c in enumerate(counts):
            it = QListWidgetItem(
                f"[{edges[i]:.4g}, {edges[i + 1]:.4g}]   {int(c)} runs   seeds {tools.seeds_in_bin(res, i)[:8]}"
            )
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.bins.addItem(it)
        v = res["values"]
        self.msg.setStyleSheet("")
        self.msg.setText(
            f"{res['stat']} of {res['measure']}: mean {v.mean():.4g}, std {v.std():.4g}, min {v.min():.4g}, max {v.max():.4g}"
        )

    def _failed(self, text: str) -> None:
        self.btn.setEnabled(True)
        self.progress.setRange(0, 1)
        self.msg.setStyleSheet("color: #e57373")
        self.msg.setText(text)

    def _bin_clicked(self, item) -> None:
        self._selected_bin = item.data(Qt.ItemDataRole.UserRole)

    def replay_seed(self) -> None:
        if self.result is None or getattr(self, "_selected_bin", None) is None:
            self.msg.setText("run first, then click a bin")
            return
        seeds = tools.seeds_in_bin(self.result, self._selected_bin)
        if not seeds:
            return
        self.ctl.seed = seeds[0]
        self.ctl.build()
        try:
            rnd = self._randomize()
        except Exception:  # noqa: BLE001
            rnd = None
        self.ctl.sim.reset(seed=seeds[0], randomize=rnd)
        self.msg.setText(f"reset to seed {seeds[0]}: press Play to watch this outcome")


class CommandPalette(QDialog):
    """Ctrl+K: type to filter every menu action and run one."""

    def __init__(self, actions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Command palette")
        self.resize(480, 360)
        self.actions_ = [a for a in actions if a.text()]
        lay = QVBoxLayout(self)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("type a command...")
        self.edit.textChanged.connect(self.fill)
        self.edit.returnPressed.connect(self.run)
        self.list = QListWidget()
        self.list.itemActivated.connect(lambda _i: self.run())
        lay.addWidget(self.edit)
        lay.addWidget(self.list)
        self.fill("")

    def fill(self, text: str) -> None:
        self.list.clear()
        words = text.lower().split()
        for a in self.actions_:
            label = a.text().replace("&", "")
            if all(w in label.lower() for w in words):
                it = QListWidgetItem(
                    label + (f"    {a.shortcut().toString()}" if not a.shortcut().isEmpty() else "")
                )
                it.setData(Qt.ItemDataRole.UserRole, a)
                self.list.addItem(it)
        if self.list.count():
            self.list.setCurrentRow(0)

    def run(self) -> None:
        it = self.list.currentItem()
        if it is not None:
            self.accept()
            it.data(Qt.ItemDataRole.UserRole).trigger()
