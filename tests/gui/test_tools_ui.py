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


def window(tmp_path):
    w = lobs.World("t", dt=0.002, substeps=5)
    w.add_box("floor", (20, 20, 0.2), fixed=True, pos=(0, 0, -0.1))
    w.add_sphere("ball", 0.1, mass=0.5, pos=(0, 0, 1.0))
    w.measure("x", lobs.props.position("ball").x)
    path = tmp_path / "t.yaml"
    Session(w).save(path)
    win = make_window(path)
    win._rebuild()
    return win


def test_console_check_compare_and_monte_carlo_panels(app, tmp_path):
    win = window(tmp_path)
    win.console.inp.setText("float(sim['x'])")
    win.console.submit()
    assert "0.0" in win.console.out.toPlainText()
    win.console.inp.setText("1/0")
    win.console.submit()
    assert "ZeroDivisionError" in win.console.out.toPlainText()
    win.check.sync = True
    win.check.start()
    assert "PASS" in win.check.out.toPlainText()
    win.compare.sync = True
    win.compare.start()
    assert "x" in win.compare.out.toPlainText() or "select" in win.compare.out.toPlainText()
    mc = win.montecarlo
    mc.sync = True
    mc.runs.setValue(6)
    mc.seconds.setValue(0.4)
    mc.random.setText('{"ball": {"pos": (Uniform(-1, 1), 0.0, 1.0)}}')
    mc.start()
    assert mc.result is not None and len(mc.result["values"]) == 6 and mc.bins.count() >= 3
    mc._bin_clicked(mc.bins.item(0))
    mc.replay_seed()
    assert "reset to seed" in mc.msg.text()
    mc.random.setText("{bad")
    mc.start()
    assert "randomize" in mc.msg.text()
    win.session.dirty = False
    win.close()


def test_command_palette_filters_and_runs_actions(app, tmp_path):
    win = window(tmp_path)
    pal = win.palette()
    n_all = pal.list.count()
    assert n_all > 10
    pal.edit.setText("frame all")
    assert pal.list.count() == 1
    win.viewport.cam.distance = 99.0
    pal.run()
    assert win.viewport.cam.distance != 99.0  # the action ran
    win.session.dirty = False
    win.close()


def test_autosave_and_recovery_flow(app, tmp_path):
    from lobs.gui.core import tools

    win = window(tmp_path)
    win.session.edit("g", lambda sp: setattr(sp, "gravity", (0.0, 0.0, -2.0)))
    win._autosave()
    assert tools.autosave_path(win.session.path).exists()
    win.save()
    assert not tools.autosave_path(win.session.path).exists()  # cleared on a real save
    win.session.dirty = False
    win.close()
