"""Outline of the world: actors (with bodies, joints, anchors, actuators), world anchors, forces, behaviors."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem


class WorldTree(QTreeWidget):
    selected = Signal(str)  # a registry path such as 'robot/arm' or 'force/gust'

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setHeaderLabels(["Name", "Kind"])
        self.setColumnWidth(0, 200)
        self._items: dict[str, QTreeWidgetItem] = {}
        self._quiet = False
        self.currentItemChanged.connect(self._changed)
        session.subscribe(lambda _kind: self.rebuild())
        self.rebuild()

    def _add(self, parent, name, kind, path):
        it = QTreeWidgetItem(parent, [name, kind])
        it.setData(0, 256, path)
        self._items[path] = it
        return it

    def rebuild(self) -> None:
        self._quiet = True
        expanded = {p for p, it in self._items.items() if it.isExpanded()}
        self.clear()
        self._items.clear()
        spec = self.session.world.spec
        root = self.invisibleRootItem()
        top = QTreeWidgetItem(root, [spec.name, "world"])
        for a in spec.actors:
            ai = self._add(top, a.name, "actor", a.name)
            for b in a.bodies:
                self._add(ai, b.name, "body", f"{a.name}/{b.name}")
            for j in a.joints:
                self._add(ai, j.name, f"joint ({j.type})", f"{a.name}/{j.name}")
            for an in a.anchors:
                self._add(ai, an.name, "anchor", f"{a.name}/{an.name}")
            for ac in a.actuators:
                self._add(ai, ac.name, "actuator", f"{a.name}/{ac.name}")
        for an in spec.anchors:
            self._add(top, an.name, "world anchor", f"world/{an.name}")
        for f in spec.forces:
            self._add(top, f.name, "adversary" if f.adversary else "force", f"force/{f.name}")
        for b in spec.behaviors:
            self._add(top, b.name, "behavior", f"behavior/{b.name}")
        for m in spec.measures:
            if not m.startswith("__"):
                self._add(top, m, "measure", f"measure/{m}")
        top.setExpanded(True)
        for p in expanded:
            if p in self._items:
                self._items[p].setExpanded(True)
        self._quiet = False

    def select_path(self, path: str) -> None:
        it = self._items.get(path)
        if it is not None:
            self._quiet = True
            self.setCurrentItem(it)
            self._quiet = False

    def _changed(self, cur, _prev) -> None:
        if cur is not None and not self._quiet:
            path = cur.data(0, 256)
            if path:
                self.selected.emit(path)
