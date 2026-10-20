"""Behavior editor: a node graph of states and transitions with action forms and a live debugger."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core import bhv
from ..core.bhv import ACTION_TYPES, PHASES

NODE_W, NODE_H = 120, 46


class StateItem(QGraphicsItem):
    def __init__(self, name: str, editor):
        super().__init__()
        self.name, self.editor = name, editor
        self.active = False
        self.initial = False
        self.breakpoint = False
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setZValue(2)

    def boundingRect(self) -> QRectF:
        return QRectF(-NODE_W / 2 - 4, -NODE_H / 2 - 4, NODE_W + 8, NODE_H + 8)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        fill = QColor(46, 125, 80) if self.active else QColor(55, 62, 76)
        border = QColor(90, 200, 255) if self.isSelected() else QColor(130, 140, 160)
        p.setBrush(QBrush(fill))
        p.setPen(QPen(border, 2.5 if self.isSelected() else 1.5))
        r = QRectF(-NODE_W / 2, -NODE_H / 2, NODE_W, NODE_H)
        p.drawRoundedRect(r, 10, 10)
        if self.initial:
            p.setPen(QPen(QColor(255, 210, 90), 2))
            p.drawRoundedRect(r.adjusted(3, 3, -3, -3), 8, 8)
        p.setPen(QColor(235, 238, 242))
        f = QFont()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        p.drawText(r, Qt.AlignmentFlag.AlignCenter, self.name)
        if self.breakpoint:
            p.setBrush(QBrush(QColor(230, 70, 70)))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(QPointF(-NODE_W / 2 + 8, -NODE_H / 2 + 8), 5, 5)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.editor.node_moved(self)
        return super().itemChange(change, value)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.editor.commit_position(self)


class TransitionItem(QGraphicsPathItem):
    def __init__(self, index: int, src: StateItem, dst: StateItem, label: str, editor):
        super().__init__()
        self.index, self.src, self.dst, self.label, self.editor = index, src, dst, label, editor
        self.firing = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(1)
        self.reroute()

    def reroute(self) -> None:
        a, b = self.src.pos(), self.dst.pos()
        path = QPainterPath()
        if self.src is self.dst:
            c = a + QPointF(0, -NODE_H / 2 - 24)
            path.addEllipse(c, 22, 22)
            self.end, self.tip = a + QPointF(10, -NODE_H / 2), QPointF(1, 1)
            self.mid = c + QPointF(0, -26)
        else:
            d = b - a
            n = math.hypot(d.x(), d.y()) or 1.0
            u = QPointF(d.x() / n, d.y() / n)
            side = QPointF(-u.y(), u.x()) * 14  # offset so A->B and B->A do not overlap
            s, t = self._edge(a + side, u), self._edge(b + side, QPointF(-u.x(), -u.y()))
            path.moveTo(s)
            path.lineTo(t)
            self.end, self.tip = t, u
            self.mid = (s + t) / 2 + side * 0.6
        self.setPath(path)

    @staticmethod
    def _edge(centre: QPointF, toward: QPointF) -> QPointF:
        sx = (NODE_W / 2) / abs(toward.x()) if abs(toward.x()) > 1e-6 else 1e9
        sy = (NODE_H / 2) / abs(toward.y()) if abs(toward.y()) > 1e-6 else 1e9
        return centre + toward * min(sx, sy)

    def boundingRect(self) -> QRectF:
        return super().boundingRect().adjusted(-80, -40, 80, 40)

    def paint(self, p: QPainter, *_):
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = (
            QColor(90, 200, 255)
            if self.isSelected()
            else QColor(120, 220, 140)
            if self.firing
            else QColor(150, 158, 172)
        )
        p.setPen(QPen(color, 2.5 if self.firing or self.isSelected() else 1.6))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(self.path())
        ang = math.atan2(self.tip.y(), self.tip.x())
        tip = self.end
        poly = QPolygonF(
            [
                tip,
                tip - QPointF(math.cos(ang - 0.4), math.sin(ang - 0.4)) * 11,
                tip - QPointF(math.cos(ang + 0.4), math.sin(ang + 0.4)) * 11,
            ]
        )
        p.setBrush(QBrush(color))
        p.drawPolygon(poly)
        p.setPen(QColor(215, 220, 228))
        f = QFont()
        f.setPointSize(8)
        p.setFont(f)
        p.drawText(self.mid + QPointF(-60, -4), self.label[:42])


class ActionRow(QWidget):
    """One action: a type chooser plus the fields that type needs."""

    FIELDS = {
        "steer": [("mode", "choice:seek,arrive,flee,intercept,hold"), ("target", "str"), ("speed", "float"), ("max_accel", "float"), ("gain", "float"), ("stop_radius", "float"), ("plane", "bool")],
        "force": [("on", "str"), ("vector", "expr"), ("max_force", "optfloat")],
        "actuate": [("actuator", "str"), ("value", "expr")],
        "set_var": [("name", "str"), ("value", "expr")],
        "emit": [("event", "str")],
        "enable_force": [("force", "str"), ("enabled", "bool")],
        "enable_behavior": [("behavior", "str"), ("enabled", "bool")],
    }  # fmt: skip
    DEFAULTS = {
        "speed": 1.0,
        "max_accel": 5.0,
        "gain": 4.0,
        "stop_radius": 0.0,
        "plane": True,
        "enabled": True,
        "mode": "seek",
    }

    def __init__(self, kind="steer", params=None, texts=None, on_change=None):
        super().__init__()
        self.on_change = on_change or (lambda: None)
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.type_box = QComboBox()
        self.type_box.addItems(list(ACTION_TYPES))
        self.type_box.setCurrentText(kind)
        self.type_box.currentTextChanged.connect(lambda k: self._build(k, {}, {}))
        self.lay.addWidget(self.type_box)
        self.editors: dict = {}
        self.holder = QWidget()
        self.hl = QHBoxLayout(self.holder)
        self.hl.setContentsMargins(0, 0, 0, 0)
        self.lay.addWidget(self.holder, 1)
        self.remove_btn = QPushButton("x")
        self.remove_btn.setMaximumWidth(26)
        self.lay.addWidget(self.remove_btn)
        self._build(kind, params or {}, texts or {})

    def _build(self, kind: str, params: dict, texts: dict) -> None:
        while self.hl.count():
            w = self.hl.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.editors = {}
        for name, typ in self.FIELDS[kind]:
            val = params.get(name, self.DEFAULTS.get(name))
            self.hl.addWidget(QLabel(name))
            if typ.startswith("choice:"):
                w = QComboBox()
                w.addItems(typ[7:].split(","))
                w.setCurrentText(str(val))
                w.currentTextChanged.connect(lambda _=0: self.on_change())
            elif typ in ("float", "optfloat"):
                w = QDoubleSpinBox()
                w.setRange(-1e9, 1e9)
                w.setDecimals(3)
                w.setValue(0.0 if val is None else float(val))
                if typ == "optfloat":
                    w.setSpecialValueText("none")
                    w.setMinimum(-1e9)
                    if val is None:
                        w.setValue(w.minimum())
                w.valueChanged.connect(lambda _=0: self.on_change())
            elif typ == "bool":
                w = QCheckBox()
                w.setChecked(bool(val))
                w.toggled.connect(lambda _=False: self.on_change())
            elif typ == "expr":
                w = QLineEdit(texts.get(name, ""))
                w.setPlaceholderText("measure expression")
                w.editingFinished.connect(self.on_change)
            else:
                w = QLineEdit("" if val is None else str(val))
                w.editingFinished.connect(self.on_change)
            self.editors[name] = (typ, w)
            self.hl.addWidget(w, 1 if typ in ("expr", "str") else 0)
        if self.isVisible() or self.parent() is not None:
            self.on_change()

    @property
    def kind(self) -> str:
        return self.type_box.currentText()

    def collect(self):
        params, texts = {}, {}
        for name, (typ, w) in self.editors.items():
            if typ.startswith("choice:"):
                params[name] = w.currentText()
            elif typ == "float":
                params[name] = w.value()
            elif typ == "optfloat":
                params[name] = None if w.value() <= w.minimum() else w.value()
            elif typ == "bool":
                params[name] = w.isChecked()
            elif typ == "expr":
                texts[name] = w.text()
            else:
                params[name] = w.text() or None
        for opt in ("on", "target", "max_force"):
            if opt in params and params[opt] is None:
                params.pop(opt)
        return self.kind, params, texts


class BehaviorEditor(QWidget):
    changed = Signal()

    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.session, self.ctl = session, ctl
        self.current = ""
        self.state_items: dict[str, StateItem] = {}
        self.trans_items: list[TransitionItem] = []
        self.breakpoints: set[tuple[str, str]] = set()
        self.link_from: StateItem | None = None
        self._own_edit = False
        self._loading = False
        root = QVBoxLayout(self)
        top = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.currentTextChanged.connect(self.load)
        self.btn_new = QPushButton("New behavior")
        self.btn_new.clicked.connect(self.new_behavior)
        self.btn_del = QPushButton("Delete behavior")
        self.btn_del.clicked.connect(self.delete_behavior)
        self.on_box = QComboBox()
        self.on_box.currentTextChanged.connect(self._set_on)
        self.latency = QDoubleSpinBox()
        self.latency.setRange(0, 60)
        self.latency.setDecimals(3)
        self.latency.setSingleStep(0.05)
        self.latency.setSuffix(" s")
        self.latency.valueChanged.connect(lambda v: self._set_attr("latency", v))
        self.period = QDoubleSpinBox()
        self.period.setRange(0, 60)
        self.period.setDecimals(3)
        self.period.setSingleStep(0.05)
        self.period.setSuffix(" s")
        self.period.valueChanged.connect(lambda v: self._set_attr("period", v))
        for w in (
            self.combo,
            self.btn_new,
            self.btn_del,
            QLabel("actor"),
            self.on_box,
            QLabel("latency"),
            self.latency,
            QLabel("period"),
            self.period,
        ):
            top.addWidget(w)
        top.addStretch()
        root.addLayout(top)
        tools = QHBoxLayout()
        for text, slot in (
            ("Add state", self.add_state),
            ("Link states", self.start_link),
            ("Delete selected", self.delete_selected),
            ("Set initial", self.set_initial),
            ("Toggle breakpoint", self.toggle_breakpoint),
            ("Auto-layout", self.auto_layout),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            tools.addWidget(b)
        tools.addStretch()
        self.live = QLabel("")
        tools.addWidget(self.live)
        root.addLayout(tools)
        split = QSplitter(Qt.Orientation.Vertical)
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-600, -400, 1200, 800)
        self.scene.selectionChanged.connect(self._selection)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setBackgroundBrush(QBrush(QColor(27, 29, 33)))
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        split.addWidget(self.view)
        self.details = QScrollArea()
        self.details.setWidgetResizable(True)
        self.detail_holder = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_holder)
        self.details.setWidget(self.detail_holder)
        split.addWidget(self.details)
        self.log = QListWidget()
        self.log.setMaximumHeight(90)
        split.addWidget(self.log)
        split.setSizes([360, 200, 70])
        root.addWidget(split)
        self.msg = QLabel("")
        root.addWidget(self.msg)
        session.subscribe(self._on_session)
        ctl.subscribe(self._on_sim)
        self.refresh_list()

    # ------------------------------------------------------------ list / load
    def _on_session(self, kind: str) -> None:
        if kind == "edit" and self._own_edit:
            return
        self.refresh_list()

    def refresh_list(self) -> None:
        spec = self.session.world.spec
        keep = self.current
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems([b.name for b in spec.behaviors])
        if keep in [b.name for b in spec.behaviors]:
            self.combo.setCurrentText(keep)
        self.combo.blockSignals(False)
        self.on_box.blockSignals(True)
        self.on_box.clear()
        self.on_box.addItems(["(global)"] + [a.name for a in spec.actors])
        self.on_box.blockSignals(False)
        self.load(self.combo.currentText())

    def spec_of(self):
        return next((b for b in self.session.world.spec.behaviors if b.name == self.current), None)

    def load(self, name: str) -> None:
        self.current = name
        self._loading = True
        self.scene.clear()
        self.state_items, self.trans_items, self.link_from = {}, [], None
        b = self.spec_of()
        has = b is not None
        for w in (self.latency, self.period, self.on_box, self.btn_del):
            w.setEnabled(has)
        self._clear_details()
        if b is None:
            self._loading = False
            return
        self.latency.setValue(b.latency)
        self.period.setValue(b.period)
        self.on_box.setCurrentText(b.on or "(global)")
        if b.machine is None:
            self.scene.addText(
                f"script behavior '{b.script.fn}' (params and actuators are edited in the inspector)"
            )
            self._loading = False
            return
        pos = bhv.positions(self.session.world.spec, name)
        for s in b.machine.states:
            it = StateItem(s.name, self)
            it.setPos(*pos[s.name])
            it.initial = s.name == (b.machine.initial or b.machine.states[0].name)
            it.breakpoint = (name, s.name) in self.breakpoints
            self.scene.addItem(it)
            self.state_items[s.name] = it
        self._draw_transitions()
        self._loading = False

    def _draw_transitions(self) -> None:
        for t in self.trans_items:
            self.scene.removeItem(t)
        self.trans_items = []
        b = self.spec_of()
        for i, t in enumerate(b.machine.transitions):
            if t.frm in self.state_items and t.to in self.state_items:
                extra = f" [{t.min_time:g}s]" if t.min_time else ""
                extra += f" p{t.priority}" if t.priority else ""
                label = bhv.transition_text(self.session.world.spec, b.name, i) + extra
                ti = TransitionItem(i, self.state_items[t.frm], self.state_items[t.to], label, self)
                self.scene.addItem(ti)
                self.trans_items.append(ti)

    def node_moved(self, _item) -> None:
        for t in self.trans_items:
            t.reroute()
            t.update()

    def commit_position(self, item: StateItem) -> None:
        if self._loading:
            return
        p = item.pos()
        name = self.current
        self._edit(
            f"move {item.name}", lambda sp: bhv.set_position(sp, name, item.name, (p.x(), p.y()))
        )

    # ------------------------------------------------------------ actions
    def _edit(self, label: str, fn, rebuild: bool = False) -> bool:
        self._own_edit = True
        try:
            self.session.edit(label, fn)
            self.msg.setStyleSheet("color: #81c784")
            self.msg.setText("")
            return True
        except Exception as e:  # noqa: BLE001 - shown to the user, the world is unchanged
            self.msg.setStyleSheet("color: #e57373")
            self.msg.setText(str(e).splitlines()[0])
            return False
        finally:
            self._own_edit = False
            if rebuild:
                self.load(self.current)
            self.changed.emit()

    def new_behavior(self) -> None:
        actor = self.on_box.currentText()
        actor = None if actor in ("(global)", "") else actor
        out = []
        if self._edit(
            "add behavior", lambda sp: out.append(bhv.add_behavior(sp, "behavior", on=actor))
        ):
            self.current = out[0]
            self.refresh_list()
            self.combo.setCurrentText(out[0])

    def delete_behavior(self) -> None:
        if self.current:
            name = self.current
            self._edit(f"delete {name}", lambda sp: bhv.delete_behavior(sp, name))
            self.current = ""
            self.refresh_list()

    def _set_on(self, text: str) -> None:
        if self._loading or not self.current:
            return
        name = self.current
        self._edit(
            "set actor",
            lambda sp: setattr(bhv.get(sp, name), "on", None if text in ("(global)", "") else text),
        )

    def _set_attr(self, attr: str, value) -> None:
        if self._loading or not self.current:
            return
        name = self.current
        self._edit(f"set {attr}", lambda sp: setattr(bhv.get(sp, name), attr, float(value)))

    def add_state(self) -> None:
        if not self.spec_of() or self.spec_of().machine is None:
            return
        name = self.current
        n = len(self.state_items)
        pos = (-150 + 120 * (n % 4), -100 + 90 * (n // 4))
        self._edit("add state", lambda sp: bhv.add_state(sp, name, "state", pos), rebuild=True)

    def delete_selected(self) -> None:
        name = self.current
        for it in self.scene.selectedItems():
            if isinstance(it, StateItem):
                st = it.name
                self._edit(
                    f"delete state {st}",
                    lambda sp, st=st: bhv.remove_state(sp, name, st),
                    rebuild=True,
                )
                return
            if isinstance(it, TransitionItem):
                i = it.index
                self._edit(
                    "delete transition",
                    lambda sp, i=i: bhv.remove_transition(sp, name, i),
                    rebuild=True,
                )
                return

    def set_initial(self) -> None:
        name = self.current
        for it in self.scene.selectedItems():
            if isinstance(it, StateItem):
                st = it.name
                self._edit(
                    "set initial",
                    lambda sp, st=st: setattr(bhv.get(sp, name).machine, "initial", st),
                    rebuild=True,
                )
                return

    def toggle_breakpoint(self) -> None:
        for it in self.scene.selectedItems():
            if isinstance(it, StateItem):
                key = (self.current, it.name)
                self.breakpoints.symmetric_difference_update({key})
                it.breakpoint = key in self.breakpoints
                it.update()

    def auto_layout(self) -> None:
        name = self.current
        n = len(self.state_items)

        def apply(sp):
            for i, s in enumerate(bhv.get(sp, name).machine.states):
                a = 2 * math.pi * i / max(n, 1)
                bhv.set_position(sp, name, s.name, (220 * math.cos(a), 160 * math.sin(a)))

        self._edit("auto-layout", apply, rebuild=True)

    def start_link(self) -> None:
        sel = [i for i in self.scene.selectedItems() if isinstance(i, StateItem)]
        if len(sel) == 2:
            self.link(sel[0], sel[1])
        elif len(sel) == 1:
            self.link(sel[0], sel[0])
        else:
            self.msg.setStyleSheet("color: #ffd166")
            self.msg.setText(
                "select the source state, then ctrl-click the target (or one state for a self loop)"
            )

    def link(self, a: StateItem, b: StateItem) -> None:
        name, frm, to = self.current, a.name, b.name
        self._edit(
            "add transition",
            lambda sp: bhv.add_transition(
                sp, name, frm, to, "ops.time() > 1.0", self.session.world
            ),
            rebuild=True,
        )

    # ------------------------------------------------------------ details
    def _clear_details(self) -> None:
        while self.detail_layout.count():
            w = self.detail_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

    def _selection(self) -> None:
        if self._loading:
            return
        try:
            sel = self.scene.selectedItems()
        except RuntimeError:  # the scene is gone while the window closes
            return
        self._clear_details()
        if len(sel) == 1 and isinstance(sel[0], StateItem):
            self._state_details(sel[0].name)
        elif len(sel) == 1 and isinstance(sel[0], TransitionItem):
            self._transition_details(sel[0].index)

    def _state_details(self, state: str) -> None:
        name = self.current
        spec = self.session.world.spec
        head = QHBoxLayout()
        ren = QLineEdit(state)
        ren.editingFinished.connect(lambda: self._rename(state, ren.text()))
        head.addWidget(QLabel("state"))
        head.addWidget(ren)
        hw = QWidget()
        hw.setLayout(head)
        self.detail_layout.addWidget(hw)
        self.phase_rows: dict[str, list[ActionRow]] = {}
        self.phase_boxes: dict[str, QVBoxLayout] = {}
        for phase in PHASES:
            self.detail_layout.addWidget(QLabel(f"<b>{phase}</b>"))
            rows: list[ActionRow] = []
            self.phase_rows[phase] = rows
            box = QVBoxLayout()
            self.phase_boxes[phase] = box
            bw = QWidget()
            bw.setLayout(box)
            for kind, params, texts in bhv.action_rows(spec, name, state, phase):
                self._add_row(box, rows, phase, state, kind, params, texts, commit=False)
            add = QPushButton(f"+ action ({phase})")
            add.clicked.connect(
                lambda _=False, box=box, rows=rows, phase=phase: self._add_row(
                    box, rows, phase, state, "steer", {}, {}, commit=True
                )
            )
            self.detail_layout.addWidget(bw)
            self.detail_layout.addWidget(add)
        self.detail_layout.addStretch()

    def _add_row(self, box, rows, phase, state, kind, params, texts, commit: bool) -> None:
        row = ActionRow(
            kind, params, texts, on_change=lambda: self._commit_actions(state, phase, rows)
        )
        row.remove_btn.clicked.connect(
            lambda: (rows.remove(row), row.deleteLater(), self._commit_actions(state, phase, rows))
        )
        rows.append(row)
        box.addWidget(row)
        if commit:
            self._commit_actions(state, phase, rows)

    def _commit_actions(self, state: str, phase: str, rows) -> None:
        name = self.current
        data = [r.collect() for r in rows if not r.isHidden()]
        world = self.session.world
        self._edit(
            f"edit {state}.{phase}", lambda sp: bhv.set_actions(sp, name, state, phase, data, world)
        )

    def _rename(self, old: str, new: str) -> None:
        new = new.strip()
        if not new or new == old:
            return
        name = self.current
        if self._edit(
            f"rename {old}", lambda sp: bhv.rename_state(sp, name, old, new), rebuild=True
        ):
            it = self.state_items.get(new)
            if it:
                it.setSelected(True)

    def _transition_details(self, i: int) -> None:
        name = self.current
        b = self.spec_of()
        t = b.machine.transitions[i]
        states = [s.name for s in b.machine.states]
        form = QFormLayout()
        frm, to = QComboBox(), QComboBox()
        frm.addItems(states)
        to.addItems(states)
        frm.setCurrentText(t.frm)
        to.setCurrentText(t.to)
        when = QLineEdit(bhv.transition_text(self.session.world.spec, name, i))
        mt = QDoubleSpinBox()
        mt.setRange(0, 1e6)
        mt.setDecimals(3)
        mt.setValue(t.min_time)
        pr = QSpinBox()
        pr.setRange(-1000, 1000)
        pr.setValue(t.priority)
        for label, w in (
            ("from", frm),
            ("to", to),
            ("when", when),
            ("min_time (s)", mt),
            ("priority", pr),
        ):
            form.addRow(label, w)
        w0 = self.session.world

        def commit(**kw) -> None:
            self._edit(
                "edit transition",
                lambda sp: bhv.update_transition(sp, name, i, w0, **kw),
                rebuild=False,
            )
            self._draw_transitions()

        frm.currentTextChanged.connect(lambda v: commit(frm=v))
        to.currentTextChanged.connect(lambda v: commit(to=v))
        when.editingFinished.connect(lambda: commit(when_text=when.text()))
        mt.valueChanged.connect(lambda v: commit(min_time=v))
        pr.valueChanged.connect(lambda v: commit(priority=v))
        holder = QWidget()
        holder.setLayout(form)
        self.detail_layout.addWidget(holder)
        self.detail_layout.addStretch()

    # ------------------------------------------------------------ live debugging
    def _on_sim(self, kind: str) -> None:
        if kind in ("invalidated",):
            self._mark(None)
        if kind not in ("stepped", "reset", "built"):
            return
        sim = self.ctl.sim
        beh = (
            sim._beh.by_name.get(self.current) if sim is not None and sim._beh is not None else None
        )
        self._mark(beh)
        if beh is not None and beh.machine is not None:
            if (
                self.ctl.playing
                and (self.current, beh.state_name) in self.breakpoints
                and self._last_state != beh.state_name
            ):
                self.ctl.pause()
                self.msg.setStyleSheet("color: #ffd166")
                self.msg.setText(f"breakpoint: entered {beh.state_name} at t={sim.time:.3f}s")
            self._last_state = beh.state_name
            self.live.setText(
                f"state {beh.state_name}   {max(0.0, sim.time - beh.enter_t):.2f}s in state"
            )
            self._fill_log(sim)

    _last_state = ""

    def _mark(self, beh) -> None:
        sim = self.ctl.sim
        for n, it in self.state_items.items():
            on = beh is not None and beh.state_name == n
            if it.active != on:
                it.active = on
                it.update()
        for t in self.trans_items:
            firing = False
            if beh is not None and sim is not None:
                key = f"__beh.{self.current}.t{t.index}.when"
                firing = bool(sim._all_values.get(key, False)) and beh.state_name == t.src.name
            if t.firing != firing:
                t.firing = firing
                t.update()
        if beh is None:
            self.live.setText("")

    def _fill_log(self, sim) -> None:
        log = [e for e in sim._beh.log if e[1] == self.current]
        if self.log.count() != len(log):
            self.log.clear()
            for t, _b, a, b in log[-30:]:
                self.log.addItem(f"{t:7.3f}s   {a} -> {b}")
            self.log.scrollToBottom()
