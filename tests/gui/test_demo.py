import numpy as np
import pytest

pytest.importorskip("PySide6")

import lobs  # noqa: E402
from lobs.gui.app import make_window  # noqa: E402
from lobs.gui.demo import build_demo, static_hint  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_demo_world_is_valid_and_visibly_moves():
    w = build_demo()
    assert w.validate().ok
    assert static_hint(w) is None
    sim = w.simulate("mujoco")
    ball0 = sim.state().body_pos[0].copy()
    sim.run(seconds=1.5)
    assert np.linalg.norm(sim.state().body_pos[0] - ball0) > 3.0  # the ball really travels
    sim.run(seconds=1.5)
    assert sim._beh.items[0].state_name == "chase"  # and the defender reacts


def test_play_in_the_gui_moves_things_and_updates_the_status(app):
    win = make_window(demo=True)
    win.resize(1200, 800)
    win._rebuild()
    before = win.viewport.grab_image().copy()
    win.ctl.speed = 0.0
    win.toggle_play()
    assert win.ctl.playing and win.a_play.text() == "Pause" and "RUNNING" in win.status_label.text()
    for _ in range(60):
        win._tick()
    assert win.ctl.sim.time > 0.2
    assert not np.array_equal(before, win.viewport.grab_image())  # pixels changed: something moved
    win.toggle_play()
    assert "paused" in win.status_label.text()
    win.session.dirty = False
    win.close()


def test_static_world_gets_a_hint(app):
    w = lobs.World("still", gravity=(0, 0, 0))
    w.add_sphere("ball", 0.1, mass=1.0, pos=(0, 0, 1))
    assert "Nothing moves" in static_hint(w)
    w2 = lobs.World("falls")
    w2.add_sphere("ball", 0.1, mass=1.0, pos=(0, 0, 1))
    assert static_hint(w2) is None  # gravity moves it
