"""Application entry point and theme."""

from __future__ import annotations

import argparse
import sys


def apply_theme(app, dark: bool = True) -> None:
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    if not dark:
        return
    pal = QPalette()
    bg, base, text, accent = (
        QColor(37, 39, 44),
        QColor(27, 29, 33),
        QColor(225, 227, 230),
        QColor(70, 140, 220),
    )
    for role, color in (
        (QPalette.ColorRole.Window, bg), (QPalette.ColorRole.WindowText, text), (QPalette.ColorRole.Base, base),
        (QPalette.ColorRole.AlternateBase, bg), (QPalette.ColorRole.Text, text), (QPalette.ColorRole.Button, bg),
        (QPalette.ColorRole.ButtonText, text), (QPalette.ColorRole.Highlight, accent),
        (QPalette.ColorRole.HighlightedText, QColor(255, 255, 255)), (QPalette.ColorRole.ToolTipBase, base),
        (QPalette.ColorRole.ToolTipText, text),
    ):  # fmt: skip
        pal.setColor(role, color)
    app.setPalette(pal)


def make_window(world_path=None):
    from .core.session import Session
    from .mainwindow import MainWindow

    session = Session.open(world_path) if world_path else Session()
    return MainWindow(session)


def run(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lobs gui", description="LoBS Studio")
    p.add_argument("world", nargs="?", help="a world file or .lobs package")
    p.add_argument("--light", action="store_true", help="use the light theme")
    args = p.parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("LoBS Studio needs PySide6 and pyqtgraph: pip install lobs[gui]", file=sys.stderr)
        return 2
    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_theme(app, dark=not args.light)
    win = make_window(args.world)
    win.show()
    return app.exec()
