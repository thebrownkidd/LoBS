"""The documented walkthrough, driven with real Qt input events on a large visible window."""

import pytest

pytest.importorskip("PySide6")

import numpy as np  # noqa: E402
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

import lobs  # noqa: E402
from lobs import ops, props  # noqa: E402
from lobs.gui.app import make_window  # noqa: E402
from lobs.gui.core import bhv, exprs  # noqa: E402
from lobs.gui.core.session import Session  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def demo_file(tmp_path):
    w = lobs.World("defence", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_box("floor", (12, 8, 0.2), fixed=True, pos=(0, 0, -0.15))
    w.add_sphere("ball", 0.12, mass=0.6, pos=(-4, 0, 1))
    w.add_box("defender", (0.5, 0.5, 1.0), mass=70.0, pos=(1, 3, 1))
    w.anchor("post", pos=(1, 3, 1))
    w.add_adversary(
        "gust", on="ball", budget=lobs.Budget(peak=1.5, impulse=0.6), vector=ops.vec(0, 1.5, 0),
        when=(props.position("ball").x > -1.0) & (props.position("ball").x < 2.0),
    )  # fmt: skip
    s = Session(w)
    s.edit("b", lambda sp: bhv.add_behavior(sp, "defender", on="defender"))
    s.edit("s", lambda sp: bhv.add_state(sp, "defender", "chase"))
    wd = s.world
    s.edit(
        "a",
        lambda sp: bhv.set_actions(
            sp,
            "defender",
            "idle",
            "do",
            [("steer", {"mode": "arrive", "target": "world/post", "speed": 2.0}, {})],
            wd,
        ),
    )
    s.edit(
        "a2",
        lambda sp: bhv.set_actions(
            sp,
            "defender",
            "chase",
            "do",
            [
                (
                    "steer",
                    {"mode": "intercept", "target": "ball", "speed": 4.0, "max_accel": 8.0},
                    {},
                )
            ],
            wd,
        ),
    )
    s.edit(
        "t",
        lambda sp: bhv.add_transition(
            sp, "defender", "idle", "chase", "rel.between(defender, ball).distance() < 3.5", wd
        ),
    )
    s.edit(
        "m", lambda sp: exprs.set_measure(sp, "gap", "rel.between(defender, ball).distance()", wd)
    )
    path = tmp_path / "defence.yaml"
    s.save(path)
    return path


def pump(win, n=30):
    for _ in range(n):
        win._tick()
        QApplication.processEvents()


def px(win, name):
    """Widget pixel of an actor centre (the image fills the widget)."""
    vp, sim = win.viewport, win.ctl.sim
    b = sim.reg.actors[sim.reg.resolve(name).index].bodies[0]
    pos = vp.picker.body_position(sim.state(), b)
    pix, _ = vp.cam.project(pos, vp.width(), vp.height())
    return QPoint(int(pix[0, 0]), int(pix[0, 1]))


def test_the_walkthrough_with_real_events(app, tmp_path):
    win = make_window(demo_file(tmp_path))
    win.resize(1900, 1000)
    win.show()
    QApplication.processEvents()
    win._rebuild()
    vp = win.viewport
    assert (
        vp.image is not None and not vp.error and vp.height() > 480
    )  # larger than the old 640x480 limit

    # 1. camera: orbit, pan, zoom by wheel
    az0, d0, look0 = vp.cam.azimuth, vp.cam.distance, vp.cam.lookat.copy()
    c = vp.rect().center()
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, c)
    QTest.mouseMove(vp, c + QPoint(60, 20))
    QTest.mouseRelease(
        vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, c + QPoint(60, 20)
    )
    assert vp.cam.azimuth != az0
    QTest.mousePress(vp, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, c)
    QTest.mouseMove(vp, c + QPoint(40, 0))
    QTest.mouseRelease(
        vp, Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, c + QPoint(40, 0)
    )
    assert not np.allclose(vp.cam.lookat, look0)
    ev = QWheelEvent(
        QPointF(c),
        QPointF(vp.mapToGlobal(c)),
        QPoint(0, 0),
        QPoint(0, 240),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(vp, ev)
    assert vp.cam.distance < d0
    QTest.keyClick(win, Qt.Key.Key_F)
    QApplication.processEvents()
    assert vp.image is not None

    # 2. click the ball to select it
    p = px(win, "ball")
    QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p)
    assert vp.selection == "ball/ball" or vp.selection.startswith("ball")

    # 3. Ctrl-drag pushes the body
    x0 = float(
        win.ctl.sim.state().body_pos[
            win.ctl.sim.reg.bodies[
                win.ctl.sim.reg.actors[win.ctl.sim.reg.resolve("ball").index].bodies[0]
            ].dyn
        ][1]
    )
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, p)
    QTest.mouseMove(vp, p + QPoint(0, -80))
    for _ in range(40):
        vp.apply_push()
        win.ctl.step(1)
    QTest.mouseRelease(
        vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, p + QPoint(0, -80)
    )
    sim = win.ctl.sim
    y1 = float(
        sim.state().body_pos[
            sim.reg.bodies[sim.reg.actors[sim.reg.resolve("ball").index].bodies[0]].dyn
        ][2]
    )
    assert abs(y1 - 1.0) > 1e-3 or x0 != 0  # the spring moved the ball

    # 4. Reset, Space to play, Space to pause, speed combo
    QTest.keyClick(win, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)
    assert win.ctl.sim.time == 0.0
    sim = win.ctl.sim
    sim.reset(initial_state={"ball": lobs.Initial(pos=(-4, 0, 1), linvel=(3.0, 0, 0))})
    win.speed_box.setCurrentIndex(0)
    assert win.ctl.speed == 0.1
    win.speed_box.setCurrentIndex(win.speed_box.count() - 1)  # max
    assert win.ctl.speed == 0.0
    QTest.keyClick(win, Qt.Key.Key_Space)
    assert win.ctl.playing
    # 5. breakpoint on chase pauses the run; the graph highlights it
    ed = win.behaviors
    ed.combo.setCurrentText("defender")
    ed.state_items["chase"].setSelected(True)
    ed.toggle_breakpoint()
    pump(win, 400)
    assert not win.ctl.playing and "breakpoint" in ed.msg.text()
    assert ed.state_items["chase"].active
    # 6. forces tab: inject a gust, the bar for the law exists
    win.forces.inject()
    assert "N s" in win.forces.msg.text()
    pump(win, 5)
    win.forces.update_live()
    assert win.forces.table.rowCount() == 1

    # 7. add a measure by typing, plot it, undo
    mp = win.measures
    mp.name_edit.setText("hh")
    mp.expr_edit.clear()
    QTest.keyClicks(mp.expr_edit, "props.height(ball)")
    QTest.keyClick(mp.expr_edit, Qt.Key.Key_Return)
    assert "hh" in win.session.world.spec.measures
    QTest.keyClick(win, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert "hh" not in win.session.world.spec.measures
    win._rebuild()

    # 8. replay last run, scrub with the slider, evaluate
    win.ctl.step(20)
    QTest.mouseClick(win.replay.last_btn, Qt.MouseButton.LeftButton)
    assert win.replay.replay is not None and win.viewport.replay is win.replay.replay
    win.replay.slider.setValue(win.replay.slider.maximum() // 2)
    assert win.viewport.image is not None
    win.replay.expr.setText("props.position(ball).x")
    QTest.keyClick(win.replay.expr, Qt.Key.Key_Return)
    assert "samples" in win.replay.msg.text()
    QTest.mouseClick(win.replay.live_btn, Qt.MouseButton.LeftButton)
    assert win.viewport.replay is None

    # 9. command palette and console typed with the keyboard
    QTest.keyClick(win, Qt.Key.Key_K, Qt.KeyboardModifier.ControlModifier)
    assert win.palette_dialog.isVisible()
    QTest.keyClicks(win.palette_dialog.edit, "frame all")
    QTest.keyClick(win.palette_dialog.edit, Qt.Key.Key_Return)
    QTest.keyClicks(win.console.inp, "float(sim['gap'])")
    QTest.keyClick(win.console.inp, Qt.Key.Key_Return)
    assert "Traceback" not in win.console.out.toPlainText()
    win.session.dirty = False
    win.close()
