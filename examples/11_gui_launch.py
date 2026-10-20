"""Open LoBS Studio on a world. Run `python examples/11_gui_launch.py --show` for the real window.

Without --show (as in the test suite) the window is built offscreen, a few ticks are simulated and the
viewport is rendered, which proves the GUI wiring without needing a display.
"""

import os
import sys
import tempfile
from pathlib import Path

import lobs
from lobs import ops, props

show = "--show" in sys.argv
if not show:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LOBS_GUI_NO_SETTINGS", "0" if show else "1")

try:
    from PySide6.QtWidgets import QApplication
except ImportError:
    print("this example needs the GUI extra: pip install 'lobs[gui]'")
    sys.exit(0)

from lobs.gui.app import apply_theme, make_window

world = lobs.World("gui-demo", dt=0.002, substeps=5)
world.add_box("floor", (8, 8, 0.2), fixed=True, pos=(0, 0, -0.1))
world.add_sphere("ball", 0.1, mass=0.5, pos=(0, 0, 1.5))
world.add_adversary("gust", on="ball", budget=lobs.Budget(peak=3.0, impulse=0.5),
                    vector=ops.vec(3.0, 0.0, 0.0), when=props.position("ball").z < 1.0)  # fmt: skip
path = Path(tempfile.mkdtemp()) / "demo.yaml"
world.save(path)

app = QApplication.instance() or QApplication([])
apply_theme(app)
win = make_window(path)
win.resize(1280, 800)
if show:
    win.show()
    sys.exit(app.exec())

win._rebuild()
win.ctl.step(100)
win.viewport.refresh()
img = win.viewport.grab_image()
print(f"rendered {img.shape[1]}x{img.shape[0]} viewport at t={win.ctl.sim.time:.2f}s; "
      f"gust impulse so far {win.ctl.sim.force_stats['gust']['impulse']:.3f} N s")  # fmt: skip
win.session.dirty = False
win.close()
