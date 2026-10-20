import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import lobs  # noqa: E402
from lobs import ops  # noqa: E402
from lobs.gui.app import make_window  # noqa: E402
from lobs.gui.core import bhv  # noqa: E402
from lobs.gui.core.session import Session  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_replay_panel_scrubs_evaluates_and_returns_to_live(app, tmp_path):
    w = lobs.World("rp", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_sphere("ball", 0.1, mass=1.0, pos=(0, 0, 1))
    w.add_adversary(
        "gust", on="ball", budget=lobs.Budget(peak=2.0, impulse=0.5), vector=ops.vec(2.0, 0.0, 0.0)
    )
    s = Session(w)
    s.edit("b", lambda sp: bhv.add_behavior(sp, "m", on="ball"))
    path = tmp_path / "rp.yaml"
    s.save(path)
    win = make_window(path)
    win.resize(900, 600)
    win._rebuild()
    rp = win.replay
    rp.from_live()
    assert "run the simulation first" in rp.msg.text()
    win.ctl.step(120)
    win._refresh_timeline()
    assert len(win.live_timeline.lanes) == 2
    rp.from_live()
    assert rp.replay is not None and win.viewport.replay is rp.replay
    rp.seek_time(0.5)
    assert abs(rp.replay.times[rp.replay.index] - 0.5) < 0.011 and "0.5" in rp.time_label.text()
    img = win.viewport.grab_image()
    assert img is not None and img.shape[2] == 3
    rp.expr.setText("props.position(ball).x")
    rp.evaluate()
    assert "samples" in rp.msg.text()
    rp.expr.setText('props.behavior_state("m")')
    rp.evaluate()
    assert "cannot be replayed" in rp.msg.text()
    rp.toggle()
    t0 = rp.replay.index
    rp._tick()
    assert rp.replay.index != t0
    rp.stop_replay()
    assert win.viewport.replay is None
    win.session.dirty = False
    win.close()
