"""Open the real (native) GUI window, drive it with timers for a few seconds and report any exception.

pixi run python tools/gui_smoke.py path/to/world.yaml
"""

import sys
import traceback

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from lobs.gui.app import apply_theme, make_window

errors = []
pos = []


def hook(kind, value, tb):
    errors.append("".join(traceback.format_exception(kind, value, tb)))


sys.excepthook = hook
app = QApplication([])
apply_theme(app)
win = make_window(sys.argv[1] if len(sys.argv) > 1 else None)
win.show()
win.showMaximized()
steps = [
    (
        1200,
        lambda: print(
            "viewport",
            win.viewport.width(),
            "x",
            win.viewport.height(),
            "window",
            win.width(),
            "x",
            win.height(),
            {k: (d.width(), d.height()) for k, d in win.docks.items() if d.isVisible()},
        ),
    ),
    (500, lambda: win.ctl.sim and win.ctl.sim.reset(seed=0)),
    (800, win.toggle_play),
    (850, lambda: pos.append(win.ctl.sim.state().body_pos[0].copy())),
    (2400, lambda: pos.append(win.ctl.sim.state().body_pos[0].copy())),
    (2500, lambda: win.behaviors.combo.setCurrentText(win.behaviors.combo.itemText(0))),
    (3500, win.toggle_play),
    (3800, win.replay.from_live),
    (4200, lambda: win.replay.slider.setValue(win.replay.slider.maximum() // 2)),
    (4800, win.replay.stop_replay),
    (5200, win.palette),
    (
        5600,
        lambda: (win.palette_dialog.close(), win.session.__setattr__("dirty", False), win.close()),
    ),
]
for ms, fn in steps:
    QTimer.singleShot(ms, fn)
code = app.exec()
print("exit", code, "frame", win.viewport.width(), "x", win.viewport.height())
print("EXCEPTIONS:", len(errors))
if len(pos) == 2:
    import numpy as np

    moved = float(np.linalg.norm(pos[1] - pos[0]))
    print("ball moved", round(moved, 2), "m while playing")
    if moved < 0.5:
        errors.append("nothing moved while playing")
for e in errors[:3]:
    print(e)
sys.exit(1 if errors else 0)
