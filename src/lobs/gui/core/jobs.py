"""Run slow library calls (import, compare, monte carlo) off the UI thread."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class Job(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, parent=None, **kwargs):
        super().__init__(parent)
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def run(self) -> None:
        try:
            self.succeeded.emit(self.fn(*self.args, **self.kwargs))
        except Exception as e:  # noqa: BLE001 - reported to the UI
            self.failed.emit(f"{type(e).__name__}: {e}")
