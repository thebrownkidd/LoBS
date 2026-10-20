"""LoBS Studio: a Qt desktop GUI over the library. ``core`` has no Qt dependency; widgets need PySide6."""

from __future__ import annotations


def main(argv=None) -> int:
    """Launch the GUI (``lobs gui [world]``)."""
    from .app import run

    return run(argv)
