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


def test_inspector_edits_flow_into_the_world_with_undo(app, tmp_path):
    win = make_window(world_file(tmp_path))
    win._rebuild()
    ins = win.inspector
    ins.show_path("ball")
    labels = [
        ins.form.itemAt(i, ins.form.ItemRole.LabelRole).widget().text()
        for i in range(ins.form.rowCount())
        if ins.form.itemAt(i, ins.form.ItemRole.LabelRole)
        and ins.form.itemAt(i, ins.form.ItemRole.LabelRole).widget()
    ]
    assert "pose.pos" in labels and "root" in labels
    ins._commit(("pose", "pos"), (0.0, 0.5, 2.0))
    assert win.session.world.spec.actors[1].pose.pos == (0.0, 0.5, 2.0)
    ins._commit(("root",), "nonsense")  # rejected by the schema, world unchanged
    assert win.session.world.spec.actors[1].root == "free"
    assert "root" in ins.status.text() or ins.status.text()
    win.session.undo()
    assert win.session.world.spec.actors[1].pose.pos == (0.0, 0.0, 1.0)
    ins.show_path("world")
    ins._commit(("gravity",), (0.0, 0.0, -3.0))
    assert win.session.world.spec.gravity == (0.0, 0.0, -3.0)
    win.add_primitive("box")
    win.add_force(False)  # nothing selected: informs instead of failing silently
    win.viewport.selection = "ball"
    win.add_force(True)
    assert win.session.world.spec.forces and win.session.world.spec.forces[0].adversary
    win.session.dirty = False
    win.close()


def defender_file(tmp_path):
    from lobs.gui.core import bhv

    w = lobs.World("def", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_sphere("ball", 0.1, mass=0.5, pos=(-3, 0, 1))
    w.add_box("defender", (0.4, 0.4, 0.8), mass=60.0, pos=(1, 2, 1))
    w.anchor("post", pos=(1, 2, 1))
    s = Session(w)
    s.edit("b", lambda sp: bhv.add_behavior(sp, "def", on="defender"))
    s.edit("s", lambda sp: bhv.add_state(sp, "def", "chase"))
    s.edit("t", lambda sp: bhv.add_transition(sp, "def", "idle", "chase", "ops.time() > 0.3", s.world))
    path = tmp_path / "def.yaml"
    s.save(path)
    return path


def test_behavior_editor_graph_edit_and_live_debug(app, tmp_path):
    win = make_window(defender_file(tmp_path))
    win.resize(1000, 700)
    win._rebuild()
    ed = win.behaviors
    ed.combo.setCurrentText("def")
    assert set(ed.state_items) == {"idle", "chase"} and len(ed.trans_items) == 1
    # editing through the widget flows into the world and survives a reload
    ed.state_items["chase"].setSelected(True)
    ed._selection()
    ed.state_items["chase"].setPos(100, 50)
    ed.commit_position(ed.state_items["chase"])
    assert win.session.world.spec.metadata["gui"]["bpos"]["def"]["chase"] == [100.0, 50.0]
    from lobs.gui.widgets.behavior_editor import ActionRow

    rows = ed.phase_rows["do"]
    ed._add_row(ed.phase_boxes["do"], rows, "do", "chase", "steer", {}, {}, commit=False)
    row = rows[-1]
    row.editors["target"][1].setText("ball")
    row.editors["speed"][1].setValue(3.0)
    ed._commit_actions("chase", "do", rows)
    act0 = win.session.world.spec.behaviors[0].machine.states[1].do[0]
    assert act0.target == "ball" and act0.speed == 3.0 and isinstance(row, ActionRow)
    # a transition edit with a bad expression is rejected and reported
    ed.trans_items[0].setSelected(True)
    ed._transition_details(0)
    ed._edit("bad", lambda sp: __import__("lobs.gui.core.bhv", fromlist=["x"]).update_transition(sp, "def", 0, win.session.world, when_text="ops.time("))
    assert ed.msg.text()
    # live: the machine moves idle -> chase at t = 0.3 and the graph shows it; breakpoint pauses the run
    ed.breakpoints.add(("def", "chase"))
    win.ctl.speed = 0.0
    win.ctl.play()
    for _ in range(400):
        win._tick()
        if not win.ctl.playing:
            break
    assert not win.ctl.playing and "breakpoint" in ed.msg.text()
    assert ed.state_items["chase"].active and not ed.state_items["idle"].active
    assert "chase" in ed.live.text() and ed.log.count() == 1
    win.session.dirty = False
    win.close()


def test_forces_panel_shows_budget_and_runtime_switch_and_gust(app, tmp_path):
    from lobs import ops
    from lobs.gui.core.session import Session as S

    w = lobs.World("f", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_box("b", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    w.add_adversary("gust", on="b", budget=lobs.Budget(peak=4.0, impulse=1.0), vector=ops.vec(4.0, 0.0, 0.0))
    path = tmp_path / "f.yaml"
    S(w).save(path)
    win = make_window(path)
    win._rebuild()
    fp = win.forces
    assert fp.table.rowCount() == 1 and fp.table.item(0, 3).text() == "adversary"
    win.ctl.step(100)
    fp.update_live()
    bar = fp.table.cellWidget(0, 4)
    assert bar.value() == 1000 and "1 / 1 N s" in bar.format()
    win.ctl.reset()
    fp.table.cellWidget(0, 0).setChecked(False)  # switch the law off at run time
    win.ctl.step(50)
    assert float(win.ctl.sim.state().body_linvel[0][0]) == 0.0
    fp.inject()
    assert "N s" in fp.msg.text()
    win.ctl.step(100)
    assert float(win.ctl.sim.state().body_linvel[0][0]) == pytest.approx(fp.impulse.value() / 1.0, rel=2e-3)
    win.session.dirty = False
    win.close()
