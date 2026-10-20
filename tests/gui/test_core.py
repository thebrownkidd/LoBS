import pytest

import lobs
from lobs.gui.core import exprs
from lobs.gui.core.session import Session
from lobs.gui.core.simctl import SimController


def box_world():
    w = lobs.World("t", dt=0.002, substeps=5)
    w.add_box("box", (0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
    return w


def test_session_save_open_roundtrip_and_dirty(tmp_path):
    s = Session(box_world())
    seen = []
    s.subscribe(seen.append)
    s.edit("gravity", lambda sp: setattr(sp, "gravity", (0.0, 0.0, -1.0)))
    assert s.dirty and s.title.endswith("*") and seen == ["edit"]
    p = s.save(tmp_path / "w.yaml")
    assert not s.dirty
    s2 = Session.open(p)
    assert s2.world.spec.gravity == (0.0, 0.0, -1.0)


def test_undo_redo_restores_exactly_and_coalesces():
    s = Session(box_world())
    before = s.world.spec.model_dump_json()
    for g in (-1.0, -2.0, -3.0):  # rapid slider-like edits coalesce into one undo step
        s.edit("gravity", lambda sp, g=g: setattr(sp, "gravity", (0.0, 0.0, g)))
    assert len(s._undo) == 1
    s.undo()
    assert s.world.spec.model_dump_json() == before
    s.redo()
    assert s.world.spec.gravity == (0.0, 0.0, -3.0)
    assert not s.can_redo()


def test_failed_edit_leaves_world_untouched():
    s = Session(box_world())
    before = s.world.spec.model_dump_json()

    def bad(sp):
        sp.gravity = (0.0, 0.0, -5.0)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        s.edit("bad", bad)
    assert s.world.spec.model_dump_json() == before and not s.dirty


def test_expressions_compile_with_positions_and_roundtrip(tmp_path):
    s = Session(box_world())
    s.edit("m", lambda sp: exprs.set_measure(sp, "h", "props.height(box)", s.world))
    assert exprs.source_of(s.world.spec, "h") == "props.height(box)"
    with pytest.raises(exprs.ExprError) as e:
        exprs.evaluate("props.height(", s.world)
    assert e.value.line == 1
    with pytest.raises(exprs.ExprError):
        exprs.evaluate("__import__('os')", s.world)
    p = s.save(tmp_path / "w.yaml")
    assert exprs.source_of(Session.open(p).world.spec, "h") == "props.height(box)"
    assert s.world.simulate("mujoco")["h"] == pytest.approx(1.0)


@pytest.mark.parametrize("be", lobs.backends.available())
def test_simctl_steps_pauses_and_resets_deterministically(be):
    s = Session(box_world())
    s.edit("m", lambda sp: exprs.set_measure(sp, "h", "props.height(box)", s.world))
    c = SimController(s)
    c.backend = be
    c.step(20)
    h1 = float(c.sim["h"])
    c.reset()
    c.step(20)
    assert float(c.sim["h"]) == h1
    c.speed = 0.0
    c.play()
    t0 = c.sim.time
    c.advance(0.05)
    assert c.sim.time > t0
    c.pause()
    t1 = c.sim.time
    assert c.advance(0.05) == 0 and c.sim.time == t1


def test_edit_invalidates_running_sim():
    s = Session(box_world())
    c = SimController(s)
    c.play()
    assert c.sim is not None
    s.edit("g", lambda sp: setattr(sp, "gravity", (0.0, 0.0, -1.0)))
    assert c.sim is None and not c.playing
