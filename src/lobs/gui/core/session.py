"""Session: the world being edited, with dirty tracking, validation and undo/redo (no Qt)."""

from __future__ import annotations

import time
from pathlib import Path

from ...schema import WorldSpec
from ...world import World

COALESCE_SECONDS = 0.6


class Session:
    def __init__(self, world: World | None = None, path: str | Path | None = None):
        self.world = world or World("untitled")
        self.path = Path(path) if path else None
        self.dirty = False
        self._undo: list[tuple[str, str]] = []  # (label, spec json before the edit)
        self._redo: list[tuple[str, str]] = []
        self._last_edit = ("", 0.0)
        self._report = None
        self._listeners: list = []

    # ------------------------------------------------------------ observers
    def subscribe(self, fn) -> None:
        """``fn(kind)`` is called after every change; kind is edit, load, undo, redo or save."""
        self._listeners.append(fn)

    def _emit(self, kind: str) -> None:
        self._report = None
        for fn in list(self._listeners):
            fn(kind)

    # ------------------------------------------------------------ io
    @classmethod
    def open(cls, path) -> Session:
        s = cls()
        s.load(path)
        return s

    def load(self, path) -> None:
        self.world = World.load(path)
        self.path = Path(path)
        self.dirty = False
        self._undo.clear()
        self._redo.clear()
        self._emit("load")

    def new(self, name: str = "untitled") -> None:
        self.world, self.path, self.dirty = World(name), None, False
        self._undo.clear()
        self._redo.clear()
        self._emit("load")

    def set_world(self, world: World, path=None, dirty: bool = False) -> None:
        """Adopt a world built elsewhere (for example by the import wizard)."""
        self.world, self.path, self.dirty = world, Path(path) if path else None, dirty
        self._undo.clear()
        self._redo.clear()
        self._emit("load")

    def save(self, path=None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path: use save(path)")
        self.world.save(target)
        self.path, self.dirty = target, False
        self._emit("save")
        return target

    @property
    def title(self) -> str:
        return f"{self.world.spec.name}{' *' if self.dirty else ''}"

    # ------------------------------------------------------------ editing
    def _snapshot(self) -> str:
        return self.world.spec.model_dump_json()

    def edit(self, label: str, fn) -> None:
        """Apply ``fn(spec)`` as one undoable edit. Rapid repeats of the same label coalesce."""
        before = self._snapshot()
        try:
            fn(self.world.spec)
        except Exception:
            self._restore(before)  # a failed edit leaves the world untouched
            raise
        now = time.monotonic()
        if not (
            self._undo
            and label == self._last_edit[0]
            and now - self._last_edit[1] < COALESCE_SECONDS
        ):
            self._undo.append((label, before))
        self._last_edit = (label, now)
        self._redo.clear()
        self.dirty = True
        self._emit("edit")

    def _restore(self, snapshot: str) -> None:
        spec = WorldSpec.model_validate_json(snapshot)
        self.world = World(spec=spec, base_dir=self.world.base_dir)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> str | None:
        if not self._undo:
            return None
        label, before = self._undo.pop()
        self._redo.append((label, self._snapshot()))
        self._restore(before)
        self.dirty = True
        self._last_edit = ("", 0.0)
        self._emit("undo")
        return label

    def redo(self) -> str | None:
        if not self._redo:
            return None
        label, after = self._redo.pop()
        self._undo.append((label, self._snapshot()))
        self._restore(after)
        self.dirty = True
        self._last_edit = ("", 0.0)
        self._emit("redo")
        return label

    # ------------------------------------------------------------ validation
    def report(self, backend: str | None = None):
        if backend is not None:
            return self.world.validate(backend)
        if self._report is None:
            self._report = self.world.validate()
        return self._report

    # ------------------------------------------------------------ gui metadata (ignored by the library)
    def gui_meta(self) -> dict:
        return self.world.spec.metadata.setdefault("gui", {})
