import pytest

import lobs
from lobs.gui.core import bhv, exprs
from lobs.gui.core.session import Session


def defence_session():
    w = lobs.World("d", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_sphere("ball", 0.1, mass=0.5, pos=(-3, 0, 1))
    w.add_box("defender", (0.4, 0.4, 0.8), mass=60.0, pos=(1, 2, 1))
    w.anchor("post", pos=(1, 2, 1))
    return Session(w)


def build_machine(s):
    s.edit("b", lambda sp: bhv.add_behavior(sp, "def", on="defender"))
    s.edit("s1", lambda sp: bhv.add_state(sp, "def", "guard"))
    s.edit("s2", lambda sp: bhv.add_state(sp, "def", "chase"))
    w = s.world
    s.edit("a1", lambda sp: bhv.set_actions(sp, "def", "guard", "do", [("steer", {"mode": "arrive", "target": "world/post", "speed": 2.0}, {})], w))
    s.edit("a2", lambda sp: bhv.set_actions(sp, "def", "chase", "do", [("steer", {"mode": "intercept", "target": "ball", "speed": 4.0, "max_accel": 8.0}, {})], w))
    s.edit("a3", lambda sp: bhv.set_actions(sp, "def", "chase", "on_enter", [("emit", {"event": "engaged"}, {})], w))
    s.edit("t1", lambda sp: bhv.add_transition(sp, "def", "guard", "chase", "rel.between(defender, ball).distance() < 5.0", w))
    s.edit("t2", lambda sp: bhv.add_transition(sp, "def", "chase", "guard", "~(rel.between(defender, ball).distance() < 7.0)", w, min_time=0.2, priority=1))


def test_edit_a_state_machine_and_run_it():
    s = defence_session()
    s.edit("clean", lambda sp: bhv.remove_state(sp, "def", "idle") if False else None)
    build_machine(s)
    b = s.world.spec.behaviors[0]
    assert [x.name for x in b.machine.states] == ["idle", "guard", "chase"]
    s.edit("init", lambda sp: setattr(sp.behaviors[0].machine, "initial", "guard"))
    s.edit("m", lambda sp: exprs.set_measure(sp, "in_chase", "props.behavior_in_state('def', 'chase')", s.world))
    assert s.report().ok, s.report().format()
    sim = s.world.simulate("mujoco", seed=0)
    sim.reset(initial_state={"ball": lobs.Initial(pos=(-3, 0, 1), linvel=(3.0, 0, 0))})
    sim.run(seconds=2.5)
    assert sim.history["in_chase"].any()  # the defender engaged the ball


def test_text_roundtrip_rename_reindex_and_remove(tmp_path):
    s = defence_session()
    build_machine(s)
    ex = s.world.spec.metadata["gui"]["expr"]
    assert "behavior/def:t0.when" in ex and "behavior/def:t1.when" in ex
    s.edit("ren", lambda sp: bhv.rename_state(sp, "def", "guard", "hold"))
    b = s.world.spec.behaviors[0]
    assert b.machine.transitions[0].frm == "hold" and b.machine.transitions[1].to == "hold"
    assert bhv.action_rows(s.world.spec, "def", "hold", "do")[0][0] == "steer"
    s.edit("rt", lambda sp: bhv.remove_transition(sp, "def", 0))
    assert len(s.world.spec.behaviors[0].machine.transitions) == 1
    assert bhv.transition_text(s.world.spec, "def", 0).startswith("~(rel.between")  # reindexed to t0
    p = s.save(tmp_path / "w.yaml")
    s2 = Session.open(p)
    assert bhv.transition_text(s2.world.spec, "def", 0).startswith("~(rel.between")
    s2.edit("rm", lambda sp: bhv.remove_state(sp, "def", "chase"))
    assert not s2.world.spec.behaviors[0].machine.transitions
    assert not any(k.startswith("behavior/def:chase.") for k in s2.world.spec.metadata["gui"]["expr"])


def test_actions_with_measures_and_validation_errors():
    s = defence_session()
    s.edit("b", lambda sp: bhv.add_behavior(sp, "b", on="defender"))
    w = s.world
    s.edit("a", lambda sp: bhv.set_actions(sp, "b", "idle", "do", [("force", {"on": None}, {"vector": "ops.vec(1.0, 0.0, 0.0)"})], w))
    rows = bhv.action_rows(s.world.spec, "b", "idle", "do")
    assert rows[0][0] == "force" and rows[0][2]["vector"] == "ops.vec(1.0, 0.0, 0.0)"
    with pytest.raises(exprs.ExprError):
        s.edit("bad", lambda sp: bhv.set_actions(sp, "b", "idle", "do", [("force", {}, {"vector": "ops.vec("})], w))
    assert len(s.world.spec.behaviors[0].machine.states[0].do) == 1  # failed edit changed nothing
    s.edit("add", lambda sp: bhv.add_transition(sp, "b", "idle", "idle", "ops.time() > 1.0", w))
    s.edit("bad2", lambda sp: bhv.add_state(sp, "b", "lonely"))
    assert any("unreachable" in i.message for i in s.report().warnings)
    positions = bhv.positions(s.world.spec, "b")
    assert set(positions) == {"idle", "lonely"}
    s.edit("pos", lambda sp: bhv.set_position(sp, "b", "idle", (10, 20)))
    assert bhv.positions(s.world.spec, "b")["idle"] == (10.0, 20.0)
