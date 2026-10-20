import pytest

import lobs
from lobs.gui.core import builder, forms
from lobs.gui.core.session import Session


def test_every_add_operation_gives_a_valid_world_and_undo_restores():
    s = Session()
    before = s.world.spec.model_dump_json()
    s.edit("ground", lambda sp: builder.add_primitive(sp, "ground"))
    for kind in ("box", "sphere", "capsule", "cylinder"):
        s.edit(f"add {kind}", lambda sp, k=kind: builder.add_primitive(sp, k, pos=(0, 0, 1)))
    s.edit("anchor", lambda sp: builder.add_world_anchor(sp))
    s.edit("force", lambda sp: builder.add_force(sp, "box"))
    s.edit("adv", lambda sp: builder.add_force(sp, "sphere", adversary=True))
    assert s.report().ok, s.report().format()
    sim = s.world.simulate("mujoco")
    sim.run(ticks=10)
    names = [a.name for a in s.world.spec.actors]
    assert names == ["ground", "box", "sphere", "capsule", "cylinder"]
    for _ in range(8):
        s._last_edit = ("", 0.0)  # no coalescing in this test
        s.undo()
    assert s.world.spec.model_dump_json() == before


def test_unique_names_duplicate_delete_and_cascade():
    s = Session()
    s.edit("a", lambda sp: builder.add_primitive(sp, "box"))
    s.edit("b", lambda sp: builder.add_primitive(sp, "box"))
    assert [a.name for a in s.world.spec.actors] == ["box", "box2"]
    s.edit("dup", lambda sp: builder.duplicate_actor(sp, "box"))
    assert s.world.spec.actors[-1].name == "box_copy"
    assert s.world.spec.actors[-1].pose.pos[0] == pytest.approx(
        s.world.spec.actors[0].pose.pos[0] + 0.5
    )
    s.edit("f", lambda sp: builder.add_force(sp, "box2"))
    s.edit("del", lambda sp: builder.delete_item(sp, "box2"))
    assert not s.world.spec.forces and s.report().ok


def test_use_current_state_as_initial():
    s = Session()
    s.edit("a", lambda sp: builder.add_primitive(sp, "sphere", pos=(0, 0, 2)))
    sim = s.world.simulate("mujoco")
    sim.run(ticks=50)
    z = float(sim.state().body_pos[0][2])
    s.edit("init", lambda sp: builder.use_current_as_initial(sp, sim))
    assert s.world.spec.actors[0].initial.pose.pos[2] == pytest.approx(z)
    assert s.world.spec.actors[0].initial.linvel[2] < 0


def test_forms_describe_force_and_actor_fields_and_assign():
    s = Session()
    s.edit("a", lambda sp: builder.add_primitive(sp, "box"))
    s.edit("f", lambda sp: builder.add_force(sp, "box", adversary=True))
    f = s.world.spec.forces[0]
    by = {x.key: x for x in forms.fields_of(f)}
    assert by["adversary"].kind == "bool" and by["vector"].kind == "measure"
    assert by["budget.peak"].kind == "float" and by["budget.peak"].optional
    assert by["on"].kind == "str"
    forms.set_path(f, ("budget", "peak"), 3.5)
    assert f.budget.peak == 3.5
    with pytest.raises(Exception):  # noqa: B017 - pydantic validation error
        forms.set_path(f, ("adversary",), "not a bool")
    actor = s.world.spec.actors[0]
    ab = {x.key: x for x in forms.fields_of(actor)}
    assert ab["root"].kind == "choice" and ab["root"].choices == ["free", "fixed"]
    assert ab["pose.pos"].kind == "vec3"
    forms.set_path(actor, ("pose", "pos"), (1.0, 2.0, 3.0))
    assert actor.pose.pos == (1.0, 2.0, 3.0)
    world_fields = {
        x.key: x for x in forms.fields_of(s.world.spec, skip=("actors", "forces", "measures"))
    }
    assert world_fields["gravity"].kind == "vec3" and world_fields["timing.dt"].kind == "float"
    assert lobs.World(spec=s.world.spec).validate().ok
