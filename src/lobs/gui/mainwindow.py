"""LoBS Studio main window: toolbar, viewport and dock panels over one Session."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSpinBox,
    QToolBar,
)

from .core import builder, tools
from .core.session import Session
from .core.simctl import SimController
from .widgets.behavior_editor import BehaviorEditor
from .widgets.forces_panel import ForcesPanel
from .widgets.import_wizard import ImportWizard
from .widgets.inspector import Inspector
from .widgets.issues import IssuesPanel
from .widgets.measures import MeasuresPanel, PlotPanel
from .widgets.replay_panel import ReplayPanel
from .widgets.timeline import Timeline
from .widgets.tools_panels import (
    CheckPanel,
    CommandPalette,
    ComparePanel,
    ConsolePanel,
    MonteCarloPanel,
)
from .widgets.viewport import Viewport
from .widgets.world_tree import WorldTree

SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 0.0]
FILTER = "Worlds (*.yaml *.yml *.json *.lobs);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self, session: Session | None = None):
        super().__init__()
        self.session = session or Session()
        self.ctl = SimController(self.session)
        import os

        self.persist = not os.environ.get("LOBS_GUI_NO_SETTINGS")
        self.settings = QSettings("lobs", "studio")
        self.viewport = Viewport(self.ctl)
        self.setCentralWidget(self.viewport)
        self.docks: dict[str, QDockWidget] = {}
        self._build_actions()
        self._build_toolbar()
        self._build_docks()
        self._build_menus()
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.status_label)
        self.session.subscribe(self._on_session)
        self.ctl.subscribe(self._on_sim)
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(30000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(300)
        self._rebuild_timer.timeout.connect(self._rebuild)
        self.setAcceptDrops(True)
        self.resize(1360, 860)
        self._on_session("load")
        self._restore()

    # ------------------------------------------------------------ construction
    def _act(self, text, slot, shortcut=None, checkable=False, checked=False) -> QAction:
        a = QAction(text, self)
        a.setCheckable(checkable)
        a.setChecked(checked)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        a.triggered.connect(slot)
        return a

    def _build_actions(self) -> None:
        self.a_new = self._act("New", self.new_world, "Ctrl+N")
        self.a_open = self._act("Open...", self.open_dialog, "Ctrl+O")
        self.a_import = self._act("Import foreign world...", lambda: self.import_dialog(), "Ctrl+I")
        self.a_save = self._act("Save", self.save, "Ctrl+S")
        self.a_save_as = self._act("Save As...", self.save_as, "Ctrl+Shift+S")
        self.a_undo = self._act("Undo", self.session.undo, "Ctrl+Z")
        self.a_redo = self._act("Redo", self.session.redo, "Ctrl+Shift+Z")
        self.a_play = self._act("Play", self.toggle_play, "Space")
        self.a_step = self._act("Step", lambda: self.ctl.step(1), "Right")
        self.a_reset = self._act("Reset", self.reset, "Ctrl+R")
        self.a_frame = self._act("Frame all", self._frame_all, "F")
        self.a_shot = self._act("Screenshot...", self.screenshot)
        self.a_forces = self._act("Force arrows", self._toggle_forces, checkable=True, checked=True)
        self.a_labels = self._act(
            "Behavior labels", self._toggle_labels, checkable=True, checked=True
        )
        self.a_follow = self._act("Follow selection", self._toggle_follow, checkable=True)
        self.a_quit = self._act("Quit", self.close, "Ctrl+Q")
        self.a_reset_layout = self._act("Reset layout", self.reset_layout)
        self.a_palette = self._act("Command palette...", self.palette, "Ctrl+K")

    def _build_toolbar(self) -> None:
        tb = QToolBar("Simulation", self)
        tb.setObjectName("sim-toolbar")
        tb.setMovable(False)
        self.addToolBar(tb)
        for a in (self.a_play, self.a_step, self.a_reset):
            tb.addAction(a)
        tb.addSeparator()
        tb.addWidget(QLabel(" speed "))
        self.speed_box = QComboBox()
        for s in SPEEDS:
            self.speed_box.addItem("max" if s == 0 else f"{s:g}x", s)
        self.speed_box.setCurrentIndex(SPEEDS.index(1.0))
        self.speed_box.currentIndexChanged.connect(
            lambda _: setattr(self.ctl, "speed", self.speed_box.currentData())
        )
        tb.addWidget(self.speed_box)
        tb.addWidget(QLabel("  backend "))
        self.backend_box = QComboBox()
        self.backend_box.addItems(self.ctl.available_backends())
        self.backend_box.currentTextChanged.connect(self._backend_changed)
        tb.addWidget(self.backend_box)
        tb.addWidget(QLabel("  seed "))
        self.seed_box = QSpinBox()
        self.seed_box.setRange(0, 2**31 - 1)
        self.seed_box.valueChanged.connect(self._seed_changed)
        tb.addWidget(self.seed_box)
        tb.addWidget(QLabel("  force scale "))
        self.fscale = QDoubleSpinBox()
        self.fscale.setDecimals(4)
        self.fscale.setRange(0.0001, 10.0)
        self.fscale.setSingleStep(0.005)
        self.fscale.setValue(self.viewport.force_scale)
        self.fscale.valueChanged.connect(lambda v: setattr(self.viewport, "force_scale", v))
        tb.addWidget(self.fscale)

    def _dock(
        self, key: str, title: str, widget, area=Qt.DockWidgetArea.LeftDockWidgetArea
    ) -> QDockWidget:
        d = QDockWidget(title, self)
        d.setObjectName(f"dock-{key}")
        scroll = (
            QScrollArea()
        )  # a scroll area has a small minimum size, so wide panels never squeeze the viewport
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(widget)
        d.setWidget(scroll)
        self.addDockWidget(area, d)
        self.docks[key] = d
        return d

    def _build_docks(self) -> None:
        self.tree = WorldTree(self.session)
        self.tree.selected.connect(self._tree_selected)
        self.issues = IssuesPanel(self.session, self.ctl)
        self._dock("outline", "Outline", self.tree)
        self.measures = MeasuresPanel(self.session, self.ctl)
        self.plots = PlotPanel(self.ctl, self.measures)
        self.plots.csv_btn.clicked.connect(self.export_csv)
        self.inspector = Inspector(self.session)
        self.tree.selected.connect(self.inspector.show_path)
        self.viewport.selected.connect(self.inspector.show_path)
        self.tree.delete_requested.connect(
            lambda p: self.session.edit(f"delete {p}", lambda sp: builder.delete_item(sp, p))
        )
        self.tree.duplicate_requested.connect(
            lambda p: self.session.edit(f"duplicate {p}", lambda sp: builder.duplicate_actor(sp, p))
        )
        self.behaviors = BehaviorEditor(self.session, self.ctl)
        self._dock(
            "behaviors", "Behavior editor", self.behaviors, Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._dock("inspector", "Inspector", self.inspector, Qt.DockWidgetArea.RightDockWidgetArea)
        self.forces = ForcesPanel(self.session, self.ctl)
        self.replay = ReplayPanel(self.ctl, self.viewport)
        self.live_timeline = Timeline()
        self.console = ConsolePanel(self.session, self.ctl)
        self.check = CheckPanel(self.session, self.ctl)
        self.compare = ComparePanel(self.session, self.ctl)
        self.montecarlo = MonteCarloPanel(self.session, self.ctl)
        for key, title, w in (
            ("console", "Console", self.console),
            ("check", "World check", self.check),
            ("compare", "Compare engines", self.compare),
            ("montecarlo", "Monte Carlo", self.montecarlo),
        ):
            self._dock(key, title, w, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock("replay", "Replay", self.replay, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock(
            "timeline", "Timeline", self.live_timeline, Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self._dock("forces", "Forces", self.forces, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock("issues", "Issues", self.issues, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock("measures", "Measures", self.measures, Qt.DockWidgetArea.BottomDockWidgetArea)
        self._dock("plots", "Plots", self.plots, Qt.DockWidgetArea.RightDockWidgetArea)
        self.tabifyDockWidget(self.docks["issues"], self.docks["measures"])
        self.tabifyDockWidget(self.docks["issues"], self.docks["forces"])
        self.tabifyDockWidget(self.docks["issues"], self.docks["timeline"])
        self.tabifyDockWidget(self.docks["issues"], self.docks["replay"])
        for key in ("console", "check", "compare", "montecarlo"):
            self.tabifyDockWidget(self.docks["issues"], self.docks[key])
        self.docks["measures"].raise_()
        self.viewport.selected.connect(self.tree.select_path)
        self.resizeDocks(
            [self.docks["behaviors"], self.docks["inspector"], self.docks["plots"]],
            [520, 520, 520],
            Qt.Orientation.Horizontal,
        )

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_sized", False):
            self._sized = True
            QTimer.singleShot(0, self._default_layout)

    def _default_layout(self) -> None:
        """Give the viewport most of the window: slim side columns and a short bottom tab area."""
        if self.persist and self.settings.value("state"):
            return
        right = [self.docks[k] for k in ("behaviors", "inspector", "plots")]
        self.resizeDocks([self.docks["outline"]], [210], Qt.Orientation.Horizontal)
        self.resizeDocks(right, [440] * 3, Qt.Orientation.Horizontal)
        self.resizeDocks(right, [320, 260, 220], Qt.Orientation.Vertical)
        self.resizeDocks([self.docks["issues"]], [210], Qt.Orientation.Vertical)
        self._default_state = self.saveState()

    def reset_layout(self) -> None:
        if getattr(self, "_default_state", None) is not None:
            self.restoreState(self._default_state)

    def _build_menus(self) -> None:
        mb = self.menuBar()
        f = mb.addMenu("&File")
        for a in (self.a_new, self.a_open, self.a_import, self.a_save, self.a_save_as):
            f.addAction(a)
        self.recent_menu = f.addMenu("Open recent")
        f.addSeparator()
        f.addAction(self.a_quit)
        e = mb.addMenu("&Edit")
        e.addAction(self.a_undo)
        e.addAction(self.a_redo)
        add = mb.addMenu("&Add")
        for kind in ("box", "sphere", "capsule", "cylinder", "ground"):
            add.addAction(kind.capitalize(), lambda k=kind: self.add_primitive(k))
        add.addSeparator()
        add.addAction(
            "World anchor",
            lambda: self.session.edit("add anchor", lambda sp: builder.add_world_anchor(sp)),
        )
        add.addAction("Force on selection", lambda: self.add_force(False))
        add.addAction("Adversary on selection", lambda: self.add_force(True))
        add.addSeparator()
        add.addAction("Use current state as initial", self.use_state_as_initial)
        r = mb.addMenu("&Run")
        for a in (self.a_play, self.a_step, self.a_reset):
            r.addAction(a)
        mb.addAction(self.a_palette)
        v = mb.addMenu("&View")
        for a in (
            self.a_frame,
            self.a_forces,
            self.a_labels,
            self.a_follow,
            self.a_shot,
            self.a_reset_layout,
        ):
            v.addAction(a)
        v.addSeparator()
        for d in self.docks.values():
            v.addAction(d.toggleViewAction())
        self._refresh_recent()

    # ------------------------------------------------------------ session / sim events
    def _on_session(self, kind: str) -> None:
        self.setWindowTitle(f"{self.session.title} - LoBS Studio")
        self.a_undo.setEnabled(self.session.can_undo())
        self.a_redo.setEnabled(self.session.can_redo())
        if kind == "load" and self.session.path:
            self._remember(self.session.path)
        if kind in ("edit", "undo", "redo", "load"):
            self._rebuild_timer.start()

    def _on_sim(self, kind: str) -> None:
        self.a_play.setText("Pause" if self.ctl.playing else "Play")

    def _rebuild(self) -> None:
        if self.ctl.build():
            self.viewport.sim_id = None
        self.viewport.refresh()
        self._status()

    def _tick(self) -> None:
        self.viewport.apply_push()
        ran = self.ctl.advance()
        if ran or self.viewport._push is not None:
            self.viewport.refresh()
            self._status()
            self._plot_ticks = getattr(self, "_plot_ticks", 0) + 1
            if self._plot_ticks % 6 == 0:
                self.plots.refresh()
                self._refresh_timeline()

    def _refresh_timeline(self) -> None:
        from .core.replay import lanes_from_sim

        sim = self.ctl.sim
        if sim is not None and len(sim.history) > 1 and self.replay.replay is None:
            self.live_timeline.set_lanes(lanes_from_sim(sim), max(sim.time, 1e-3))
            self.live_timeline.set_playhead(sim.time)

    def _status(self) -> None:
        sim = self.ctl.sim
        if sim is None:
            self.status_label.setText(self.ctl.error or "no simulation")
            return
        rep = self.session.report()
        self.status_label.setText(
            f"t={sim.time:.2f}s  {self.ctl.realtime_factor:4.1f}x realtime  "
            f"{len(rep.errors)} errors, {len(rep.warnings)} warnings"
        )

    # ------------------------------------------------------------ actions
    def add_primitive(self, kind: str) -> None:
        self.session.edit(
            f"add {kind}", lambda sp: builder.add_primitive(sp, kind, fixed=kind == "ground")
        )

    def add_force(self, adversary: bool) -> None:
        sel = (
            self.viewport.selection.split("/")[0]
            if self.viewport.selection
            else (self.inspector.path or "")
        )
        names = [a.name for a in self.session.world.spec.actors]
        if sel not in names:
            self.statusBar().showMessage("Select an actor first to attach a force to it", 4000)
            return
        self.session.edit("add force", lambda sp: builder.add_force(sp, sel, adversary=adversary))

    def use_state_as_initial(self) -> None:
        sim = self.ctl.sim
        if sim is None:
            return
        self.session.edit(
            "use state as initial", lambda sp: builder.use_current_as_initial(sp, sim)
        )

    def palette(self) -> CommandPalette:
        acts = [
            a
            for a in self.findChildren(QAction)
            if a.text() and a.menu() is None and not a.isSeparator()
        ]
        pal = CommandPalette(acts, self)
        pal.show()
        self.palette_dialog = pal
        return pal

    def _autosave(self) -> None:
        try:
            tools.write_autosave(self.session)
        except Exception:  # noqa: BLE001 - autosave must never interrupt work
            pass

    def toggle_play(self) -> None:
        if self.ctl.playing:
            self.ctl.pause()
        else:
            self.ctl.play()

    def reset(self) -> None:
        self.ctl.reset()
        self.viewport.refresh()

    def new_world(self) -> None:
        if self._confirm_discard():
            self.session.new()

    def open_dialog(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open world", self._last_dir(), FILTER)
        if path:
            self.open_path(path)

    def open_path(self, path, recover: bool = True) -> None:
        try:
            auto = tools.recoverable(path) if recover else None
            if auto is not None:
                ask = QMessageBox.question(
                    self,
                    "Recover",
                    f"An autosave newer than {Path(path).name} exists. Open the autosave instead?",
                )
                if ask == QMessageBox.StandardButton.Yes:
                    self.session.load(auto)
                    self.session.path = Path(path)
                    self.session.dirty = True
                    return
            self.session.load(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot open", str(e))

    def import_dialog(self, source: str | None = None) -> ImportWizard | None:
        if not self._confirm_discard():
            return None
        wiz = ImportWizard(self, source)
        wiz.imported.connect(self._adopt_import)
        wiz.show()
        self.wizard = wiz
        return wiz

    def _adopt_import(self, world, out: str) -> None:
        path = Path(out) / "world.yaml" if out else None
        self.session.set_world(world, path=path)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix in (".yaml", ".yml", ".json", ".lobs") or path.is_dir():
                if self._confirm_discard():
                    self.open_path(path)
            else:
                self.import_dialog(str(path))
            break

    def save(self) -> None:
        if self.session.path is None:
            self.save_as()
            return
        self._save_to(self.session.path)

    def save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save world", self._last_dir(), "World (*.yaml)"
        )
        if path:
            self._save_to(path)

    def _save_to(self, path) -> None:
        try:
            self.session.save(path)
            tools.clear_autosave(path)
            self._remember(path)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Cannot save", str(e))

    def export_csv(self) -> None:
        from .core.plotdata import export_csv

        sim = self.ctl.sim
        names = sorted(self.measures.plotted) or [n for n in sim.history.names()] if sim else []
        if sim is None or not names:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measures", self._last_dir(), "CSV (*.csv)"
        )
        if path:
            export_csv(sim.history, names, path)

    def screenshot(self) -> None:
        from ..viz import write_png

        img = self.viewport.grab_image()
        if img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save screenshot", self._last_dir(), "PNG (*.png)"
        )
        if path:
            write_png(path, img)

    def _confirm_discard(self) -> bool:
        if not self.session.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Discard unsaved changes?")
        return r == QMessageBox.StandardButton.Yes

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        if self.persist:
            self.settings.setValue("geometry", self.saveGeometry())
            self.settings.setValue("state", self.saveState())
        self.timer.stop()
        self.ctl.close()
        e.accept()

    # ------------------------------------------------------------ view toggles
    def _frame_all(self) -> None:
        self.viewport.frame_all()
        self.viewport.refresh()

    def _toggle_forces(self) -> None:
        self.viewport.show_forces = self.a_forces.isChecked()
        self.viewport.update()

    def _toggle_labels(self) -> None:
        self.viewport.show_labels = self.a_labels.isChecked()
        self.viewport.update()

    def _toggle_follow(self) -> None:
        self.viewport.follow = self.a_follow.isChecked()

    def _backend_changed(self, name: str) -> None:
        if name and name != self.ctl.backend:
            self.ctl.set_backend(name)
            self.viewport.sim_id = None
            self.viewport.refresh()

    def _seed_changed(self, v: int) -> None:
        self.ctl.seed = v
        self.ctl.reset()
        self.viewport.refresh()

    def _tree_selected(self, path: str) -> None:
        if path.startswith("behavior/"):
            self.behaviors.combo.setCurrentText(path.split("/", 1)[1])
            self.docks["behaviors"].raise_()
        if path and self.ctl.sim is not None:
            try:
                self.viewport.selection = path
                self.viewport.update()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------ settings / recents
    def _recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        return [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])

    def _remember(self, path) -> None:
        paths = [str(Path(path).resolve())] + [
            p for p in self._recent() if p != str(Path(path).resolve())
        ]
        self.settings.setValue("recent", paths[:8])
        self.settings.setValue("last_dir", str(Path(path).resolve().parent))
        self._refresh_recent()

    def _refresh_recent(self) -> None:
        self.recent_menu.clear()
        for p in self._recent():
            a = QAction(p, self)
            a.triggered.connect(lambda _=False, p=p: self._confirm_discard() and self.open_path(p))
            self.recent_menu.addAction(a)

    def _last_dir(self) -> str:
        return str(self.settings.value("last_dir", str(Path.cwd())))

    def _restore(self) -> None:
        if not self.persist:
            return
        g, s = self.settings.value("geometry"), self.settings.value("state")
        if g:
            self.restoreGeometry(g)
        if s:
            self.restoreState(s)
