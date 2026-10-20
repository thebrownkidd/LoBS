"""Validation issues for the current world and backend (clickable)."""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QListWidget, QListWidgetItem


class IssuesPanel(QListWidget):
    def __init__(self, session, ctl, parent=None):
        super().__init__(parent)
        self.session, self.ctl = session, ctl
        session.subscribe(lambda _kind: self.refresh())
        ctl.subscribe(lambda kind: self.refresh() if kind == "built" else None)
        self.refresh()

    def refresh(self) -> None:
        self.clear()
        rep = self.session.report(self.ctl.backend)
        for i in rep.issues:
            it = QListWidgetItem(f"[{i.level}] {i.path}: {i.message}")
            it.setForeground(QColor(230, 110, 110) if i.level == "error" else QColor(225, 180, 80))
            self.addItem(it)
        if self.ctl.error and not rep.errors:
            self.addItem(QListWidgetItem(f"[error] simulation: {self.ctl.error}"))
        if self.count() == 0:
            self.addItem(QListWidgetItem("No issues"))
