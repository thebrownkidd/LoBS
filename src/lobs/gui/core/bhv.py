"""Behavior editing operations (no Qt): states, transitions and actions, with their expression texts.

Measure-valued fields are authored as expression text; the text is kept in ``metadata['gui']['expr']`` under
``behavior/<name>:<key>`` where ``<key>`` is the same key ``BehaviorSpec.nodes()`` uses, so it round-trips."""

from __future__ import annotations

import math

from ...errors import LobsError
from ...schema import behavior as S
from . import exprs

ACTION_TYPES = {
    "steer": S.Steer, "force": S.Force, "actuate": S.Actuate, "set_var": S.SetVar,
    "emit": S.Emit, "enable_force": S.EnableForce, "enable_behavior": S.EnableBehavior,
}  # fmt: skip
MEASURE_FIELD = {"actuate": "value", "set_var": "value", "force": "vector"}
PHASES = ("on_enter", "do", "on_exit")


def _exprs(spec) -> dict:
    return spec.metadata.setdefault("gui", {}).setdefault("expr", {})


def meta_key(bname: str, key: str) -> str:
    return f"behavior/{bname}:{key}"


def get(spec, name: str) -> S.BehaviorSpec:
    b = next((b for b in spec.behaviors if b.name == name), None)
    if b is None:
        raise LobsError(f"no behavior '{name}'")
    return b


def unique(existing, base: str) -> str:
    names, i, cand = set(existing), 2, base
    while cand in names:
        cand, i = f"{base}{i}", i + 1
    return cand


# ------------------------------------------------------------ behaviors
def add_behavior(
    spec, name: str | None = None, on: str | None = None, script: str | None = None
) -> str:
    name = unique([b.name for b in spec.behaviors], name or "behavior")
    if script is not None:
        b = S.BehaviorSpec(name=name, on=on, script=S.ScriptSpec(fn=script))
    else:
        m = S.StateMachineSpec(states=[S.StateSpec(name="idle")], initial="idle")
        b = S.BehaviorSpec(name=name, on=on, machine=m)
    spec.behaviors.append(b)
    return name


def delete_behavior(spec, name: str) -> None:
    spec.behaviors = [b for b in spec.behaviors if b.name != name]
    ex = _exprs(spec)
    for k in [k for k in ex if k.startswith(f"behavior/{name}:")]:
        del ex[k]
    pos = spec.metadata.get("gui", {}).get("bpos", {})
    pos.pop(name, None)


# ------------------------------------------------------------ states
def add_state(spec, bname: str, name: str | None = None, pos=None) -> str:
    b = get(spec, bname)
    name = unique([s.name for s in b.machine.states], name or "state")
    b.machine.states.append(S.StateSpec(name=name))
    if b.machine.initial is None:
        b.machine.initial = name
    if pos is not None:
        set_position(spec, bname, name, pos)
    return name


def remove_state(spec, bname: str, name: str) -> None:
    b = get(spec, bname)
    m = b.machine
    m.states = [s for s in m.states if s.name != name]
    keep = [t for t in m.transitions if name not in (t.frm, t.to)]
    texts = [transition_text(spec, bname, i) for i, t in enumerate(m.transitions) if t in keep]
    set_transitions(spec, bname, keep, texts)
    if m.initial == name:
        m.initial = m.states[0].name if m.states else None
    spec.metadata.get("gui", {}).get("bpos", {}).get(bname, {}).pop(name, None)
    _purge_state_texts(spec, bname, name)


def _purge_state_texts(spec, bname, state) -> None:
    ex = _exprs(spec)
    for k in [k for k in ex if k.startswith(f"behavior/{bname}:{state}.")]:
        del ex[k]


def rename_state(spec, bname: str, old: str, new: str) -> None:
    b = get(spec, bname)
    m = b.machine
    if new in [s.name for s in m.states]:
        raise LobsError(f"state '{new}' already exists")
    for s in m.states:
        if s.name == old:
            s.name = new
    for t in m.transitions:
        t.frm = new if t.frm == old else t.frm
        t.to = new if t.to == old else t.to
    if m.initial == old:
        m.initial = new
    ex = _exprs(spec)
    for k in [k for k in ex if k.startswith(f"behavior/{bname}:{old}.")]:
        ex[k.replace(f":{old}.", f":{new}.", 1)] = ex.pop(k)
    bp = spec.metadata.get("gui", {}).get("bpos", {}).get(bname, {})
    if old in bp:
        bp[new] = bp.pop(old)


def set_position(spec, bname: str, state: str, pos) -> None:
    spec.metadata.setdefault("gui", {}).setdefault("bpos", {}).setdefault(bname, {})[state] = [
        float(pos[0]),
        float(pos[1]),
    ]


def positions(spec, bname: str) -> dict:
    """Stored node positions; unplaced states are laid out on a circle."""
    b = get(spec, bname)
    stored = spec.metadata.get("gui", {}).get("bpos", {}).get(bname, {})
    n = max(len(b.machine.states), 1)
    out = {}
    for i, s in enumerate(b.machine.states):
        if s.name in stored:
            out[s.name] = tuple(stored[s.name])
        else:
            a = 2 * math.pi * i / n
            out[s.name] = (220 * math.cos(a), 160 * math.sin(a))
    return out


# ------------------------------------------------------------ actions
def _build_action(kind: str, params: dict, texts: dict, world):
    cls = ACTION_TYPES[kind]
    params = dict(params)
    mf = MEASURE_FIELD.get(kind)
    if mf is not None:
        params[mf] = exprs.evaluate(texts.get(mf, ""), world).node
    return cls(**params)


def set_actions(spec, bname: str, state: str, phase: str, rows, world) -> None:
    """``rows`` is a list of ``(kind, params, texts)``; replaces the actions of one state phase."""
    if phase not in PHASES:
        raise LobsError(f"unknown phase '{phase}'")
    b = get(spec, bname)
    st = next((s for s in b.machine.states if s.name == state), None)
    if st is None:
        raise LobsError(f"no state '{state}'")
    built = [_build_action(k, p, t, world) for k, p, t in rows]
    ex = _exprs(spec)
    for k in [k for k in ex if k.startswith(f"behavior/{bname}:{state}.{phase}.")]:
        del ex[k]
    for i, (kind, _, texts) in enumerate(rows):
        mf = MEASURE_FIELD.get(kind)
        if mf is not None:
            ex[meta_key(bname, f"{state}.{phase}.{i}.{mf}")] = texts[mf].strip()
    setattr(st, phase, built)


def action_rows(spec, bname: str, state: str, phase: str):
    """Current actions as ``[(kind, params, texts)]`` for editing."""
    b = get(spec, bname)
    st = next(s for s in b.machine.states if s.name == state)
    ex = _exprs(spec)
    rows = []
    for i, a in enumerate(getattr(st, phase)):
        kind = a.do
        params = a.model_dump(
            exclude={"do", *([MEASURE_FIELD[kind]] if kind in MEASURE_FIELD else [])}
        )
        texts = {}
        mf = MEASURE_FIELD.get(kind)
        if mf is not None:
            key = meta_key(bname, f"{state}.{phase}.{i}.{mf}")
            texts[mf] = ex.get(key) or getattr(a, mf).model_dump_json()
        rows.append((kind, params, texts))
    return rows


# ------------------------------------------------------------ transitions
def transition_text(spec, bname: str, i: int) -> str:
    t = get(spec, bname).machine.transitions[i]
    return _exprs(spec).get(meta_key(bname, f"t{i}.when")) or t.when.model_dump_json()


def set_transitions(spec, bname: str, transitions, texts) -> None:
    """Replace all transitions (with their ``when`` texts) and rewrite the stored expression keys."""
    b = get(spec, bname)
    ex = _exprs(spec)
    for k in [k for k in ex if k.startswith(f"behavior/{bname}:t") and k.endswith(".when")]:
        del ex[k]
    b.machine.transitions = list(transitions)
    for i, text in enumerate(texts):
        if text and not text.lstrip().startswith("{"):
            ex[meta_key(bname, f"t{i}.when")] = text.strip()


def add_transition(
    spec, bname: str, frm: str, to: str, when_text: str, world, min_time=0.0, priority=0
) -> None:
    b = get(spec, bname)
    names = [s.name for s in b.machine.states]
    if frm not in names or to not in names:
        raise LobsError("transition endpoints must be existing states")
    node = exprs.evaluate(when_text, world).node
    texts = [transition_text(spec, bname, i) for i in range(len(b.machine.transitions))] + [
        when_text
    ]
    t = S.TransitionSpec(frm=frm, to=to, when=node, min_time=min_time, priority=priority)
    set_transitions(spec, bname, [*b.machine.transitions, t], texts)


def update_transition(
    spec,
    bname: str,
    i: int,
    world,
    *,
    frm=None,
    to=None,
    when_text=None,
    min_time=None,
    priority=None,
) -> None:
    b = get(spec, bname)
    ts = list(b.machine.transitions)
    texts = [transition_text(spec, bname, j) for j in range(len(ts))]
    t = ts[i].model_copy()
    if frm is not None:
        t.frm = frm
    if to is not None:
        t.to = to
    if when_text is not None:
        t.when = exprs.evaluate(when_text, world).node
        texts[i] = when_text
    if min_time is not None:
        t.min_time = float(min_time)
    if priority is not None:
        t.priority = int(priority)
    ts[i] = t
    set_transitions(spec, bname, ts, texts)


def remove_transition(spec, bname: str, i: int) -> None:
    b = get(spec, bname)
    ts = list(b.machine.transitions)
    texts = [transition_text(spec, bname, j) for j in range(len(ts))]
    del ts[i], texts[i]
    set_transitions(spec, bname, ts, texts)
