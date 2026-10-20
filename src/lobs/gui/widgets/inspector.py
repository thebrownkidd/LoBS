"""Inspector: a schema-driven property form for whatever is selected in the outline."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSpinBox,
    QWidget,
)

from ..core import builder, exprs, forms

SKIP = {
    "world": (
        "actors",
        "anchors",
        "forces",
        "behaviors",
        "measures",
        "imports",
        "metadata",
        "georeference",
        "backend_options",
        "contact",
        "lobs_schema",
    ),
    "actor": ("bodies", "joints", "actuators", "anchors", "metadata", "source", "initial", "tags"),
    "body": ("geoms", "source", "tags"),
}
BIG = 1e9


class _Vec(QWidget):
    def __init__(self, n: int, values, on_change):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.boxes = []
        for v in values:
            b = QDoubleSpinBox()
            b.setRange(-BIG, BIG)
            b.setDecimals(4)
            b.setSingleStep(0.1)
            b.setValue(float(v))
            b.valueChanged.connect(lambda _=0: on_change(tuple(x.value() for x in self.boxes)))
            self.boxes.append(b)
            lay.addWidget(b)


class Inspector(QScrollArea):
    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.path = ""
        self._own_edit = False
        self.setWidgetResizable(True)
        self.holder = QWidget()
        self.form = QFormLayout(self.holder)
        self.setWidget(self.holder)
        self.status = QLabel("")
        session.subscribe(self._on_session)
        self.show_path("world")

    def _on_session(self, kind: str) -> None:
        if kind == "edit" and self._own_edit:
            return
        self.show_path(self.path or "world")

    # ------------------------------------------------------------ building the form
    def show_path(self, path: str) -> None:
        self.path = path
        while self.form.rowCount():
            self.form.removeRow(0)
        obj, kind = builder.resolve_object(self.session.world.spec, path)
        if obj is None:
            self.form.addRow(QLabel(f"{path or 'nothing'} has no editable properties here"))
            return
        if kind == "behavior":
            self.form.addRow(QLabel("Edit this behavior in the Behavior editor"))
        self.form.addRow(QLabel(f"<b>{path or 'world'}</b> ({kind})"))
        for f in forms.fields_of(obj, skip=SKIP.get(kind, ())):
            if f.kind == "json":
                continue
            self.form.addRow(self._label(f), self._editor(f, kind))
        self.status = QLabel("")
        self.form.addRow(self.status)

    def _label(self, f: forms.FieldSpec) -> QLabel:
        lab = QLabel(f.label)
        if f.description:
            lab.setToolTip(f.description)
        return lab

    def _editor(self, f: forms.FieldSpec, kind: str) -> QWidget:
        key = f.path
        if f.kind == "float":
            return self._optional(
                f, lambda v: self._commit(key, v), self._spin(f.value or 0.0, key)
            )
        if f.kind == "int":
            b = QSpinBox()
            b.setRange(-(2**30), 2**30)
            b.setValue(int(f.value or 0))
            b.valueChanged.connect(lambda v: self._commit(key, v))
            return b
        if f.kind == "bool":
            c = QCheckBox()
            c.setChecked(bool(f.value))
            c.toggled.connect(lambda v: self._commit(key, v))
            return c
        if f.kind == "str":
            e = QLineEdit("" if f.value is None else str(f.value))
            e.editingFinished.connect(
                lambda e=e: self._commit(key, e.text() or (None if f.optional else ""))
            )
            return e
        if f.kind == "choice":
            c = QComboBox()
            c.addItems([str(x) for x in f.choices])
            c.setCurrentText(str(f.value))
            c.currentTextChanged.connect(lambda v: self._commit(key, v))
            return c
        if f.kind in ("vec2", "vec3", "vec4"):
            n = int(f.kind[-1])
            return _Vec(
                n, f.value if f.value is not None else (0.0,) * n, lambda v: self._commit(key, v)
            )
        if f.kind == "measure":
            return self._measure_editor(f, kind)
        if f.kind == "union":
            return QLabel(str(f.value))
        return QLabel("")

    def _spin(self, value, key):
        b = QDoubleSpinBox()
        b.setRange(-BIG, BIG)
        b.setDecimals(5)
        b.setSingleStep(0.1)
        b.setValue(float(value))
        b.valueChanged.connect(lambda v: self._commit(key, v))
        return b

    def _optional(self, f, _commit, spin: QDoubleSpinBox) -> QWidget:
        if not f.optional:
            return spin
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        chk = QCheckBox()
        chk.setChecked(f.value is not None)
        spin.setEnabled(f.value is not None)

        def toggled(on: bool) -> None:
            spin.setEnabled(on)
            self._commit(f.path, spin.value() if on else None)

        chk.toggled.connect(toggled)
        lay.addWidget(chk)
        lay.addWidget(spin)
        return w

    def _measure_editor(self, f: forms.FieldSpec, kind: str) -> QWidget:
        spec = self.session.world.spec
        meta_key = f"{self.path}.{f.key}"
        text = spec.metadata.get("gui", {}).get("expr", {}).get(meta_key)
        node = f.value
        e = QLineEdit(
            text if text is not None else ("" if node is None else node.model_dump_json())
        )
        e.setToolTip("A measure expression, e.g. props.velocity(ball) * -2.0, or empty for none")

        def done() -> None:
            t = e.text().strip()
            if t.startswith("{"):
                return  # raw JSON node: edit via an expression instead
            self._commit_measure(f.path, meta_key, t, e)

        e.editingFinished.connect(done)
        return e

    # ------------------------------------------------------------ commits
    def _commit(self, path: tuple, value) -> None:
        target = self.path

        def apply(spec):
            obj, _ = builder.resolve_object(spec, target)
            forms.set_path(obj, path, value)

        self._edit(f"set {target}.{'.'.join(path)}", apply)

    def _commit_measure(self, path: tuple, meta_key: str, text: str, edit: QLineEdit) -> None:
        target = self.path
        try:
            node = exprs.evaluate(text, self.session.world).node if text else None
        except exprs.ExprError as err:
            self.status.setStyleSheet("color: #e57373")
            self.status.setText(f"{path[-1]}: {err}")
            edit.setStyleSheet("border: 1px solid #e57373")
            return
        edit.setStyleSheet("")

        def apply(spec):
            obj, _ = builder.resolve_object(spec, target)
            forms.set_path(obj, path, node)
            expr = spec.metadata.setdefault("gui", {}).setdefault("expr", {})
            if text:
                expr[meta_key] = text
            else:
                expr.pop(meta_key, None)

        self._edit(f"set {target}.{'.'.join(path)}", apply)

    def _edit(self, label: str, fn) -> None:
        self._own_edit = True
        try:
            self.session.edit(label, fn)
            self.status.setStyleSheet("color: #81c784")
            self.status.setText("")
        except Exception as err:  # noqa: BLE001 - validation errors are shown, the world is unchanged
            self.status.setStyleSheet("color: #e57373")
            self.status.setText(str(err).splitlines()[0])
        finally:
            self._own_edit = False
