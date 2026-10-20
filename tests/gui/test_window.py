import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import lobs  # noqa: E402
from lobs.gui.app import make_window  # noqa: E402
from lobs.gui.core.session import Session  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def world_file(tmp_path):
    w = lobs.World("demo", dt=0.002, substeps=5)
    w.add_box("floor", (6, 6, 0.2), fixed=True, pos=(0, 0, -0.1))
    w.add_sphere("ball", 0.1, mass=0.5, pos=(0, 0, 1.0))
    w.anchor("goal", pos=(1, 0, 1))
    path = tmp_path / "demo.yaml"
    w.save(path)
    return path


def test_window_opens_renders_plays_and_edits(app, tmp_path):
    from lobs.gui.mainwindow import MainWindow

    win = make_window(world_file(tmp_path))
    assert isinstance(win, MainWindow)
    win.resize(640, 480)
    win._rebuild()
    assert win.viewport.image is not None and win.viewport.image.width() > 0
    assert win.tree.topLevelItemCount() == 1
    assert "ball" in win.tree._items
    t0 = win.ctl.sim.time
    win.ctl.speed = 0.0
    win.ctl.play()
    win._tick()
    assert win.ctl.sim.time > t0
    win.ctl.pause()
    win.session.edit("g", lambda sp: setattr(sp, "gravity", (0.0, 0.0, -1.0)))
    assert win.windowTitle().startswith("demo *")
    assert win.ctl.sim is None
    win._rebuild()
    assert win.viewport.grab_image().shape[2] == 3
    win.session.dirty = False
    win.close()


def test_picking_selects_the_ball(app, tmp_path):
    win = make_window(world_file(tmp_path))
    win.resize(640, 480)
    win._rebuild()
    vp, sim = win.viewport, win.ctl.sim
    body = sim.reg.resolve("ball/ball" if False else "ball")
    b = sim.reg.actors[body.index].bodies[0]
    centre = vp.picker.body_position(sim.state(), b)
    pix, _ = vp.cam.project(centre, vp.width(), vp.height())
    assert vp.picker.pick(vp.cam, pix[0, 0], pix[0, 1], vp.width(), vp.height(), sim.state()) == b
    assert vp.picker.pick(vp.cam, 2.0, 2.0, vp.width(), vp.height(), sim.state()) is None
    win.session.dirty = False
    win.close()


def test_cli_gui_subcommand_is_wired():
    import lobs.cli as cli

    assert callable(cli.cmd_gui)
    assert Session  # core imports without a display


def test_measure_panel_adds_validates_and_plots(app, tmp_path):
    win = make_window(world_file(tmp_path))
    win.resize(900, 600)
    win._rebuild()
    mp = win.measures
    mp.name_edit.setText("h")
    mp.expr_edit.setText("props.height(")
    mp.add()
    assert "h" not in win.session.world.spec.measures and "line 1" in mp.msg.text()
    mp.expr_edit.setText("props.height(ball)")
    mp.add()
    assert "h" in win.session.world.spec.measures
    win._rebuild()
    mp.plotted.add("h")
    win.ctl.step(30)
    win.plots.refresh()
    assert "h" in win.plots.curves
    mp.update_values()
    row = [r for r in range(mp.table.rowCount()) if mp.table.item(r, 1).text() == "h"][0]
    assert float(mp.table.item(row, 2).text()) < 1.0
    win.session.dirty = False
    win.close()
