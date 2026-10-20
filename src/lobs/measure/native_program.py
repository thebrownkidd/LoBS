"""Export a compiled measure graph to the native interpreter (src/native/program.hpp).

Every node either maps to a native opcode or becomes a callback into its original Python closure, so the
native program is always a complete, equivalent evaluator; coverage only decides how much of it is fast."""

from __future__ import annotations

import numpy as np

from .. import native
from ..schema.measure import ConstNode, OpNode, PropNode, RelNode, TimeNode
from .units import RADIAN

# opcode numbers must match enum Op in program.hpp
OPS = {
    "CONST": 0, "TIME": 1, "ADD": 2, "SUB": 3, "MUL": 4, "DIV": 5, "POW": 6, "NEG": 7, "ABS": 8, "LT": 9, "LE": 10,
    "GT": 11, "GE": 12, "EQ": 13, "AND": 14, "OR": 15, "NOT": 16, "INDEX": 17, "NORM": 18, "DOT": 19, "CROSS": 20,
    "SIN": 21, "COS": 22, "CLIP": 23, "WHERE": 24, "MAXOF": 25, "MINOF": 26, "MEANOF": 27, "SUMOF": 28, "VEC": 29,
    "UNIT": 30, "CLIPNORM": 31, "COPY": 32, "DERIV": 33, "INTEGRAL": 34, "RUNMAX": 35, "RUNMIN": 36, "ATRESET": 37,
    "SINCERESET": 38, "WINDOW": 39, "DURTRUE": 40, "TIMESINCE": 41, "RISING": 42, "FALLING": 43, "LATCH": 44,
    "HOLD": 45, "COOLDOWN": 46, "ONCE": 47, "SAMPLEON": 48, "P_POS": 49, "P_VEL": 50, "P_SPEED": 51, "P_HEIGHT": 52,
    "P_ORIENT": 53, "P_ANGVEL": 54, "P_SPIN": 55, "P_JQ": 56, "P_JQD": 57, "P_JEFF": 58, "REL_DIST": 59,
    "P_EVENT": 60, "P_BSTATE": 61, "P_BINSTATE": 62, "P_BTIME": 63, "P_BVAR": 64, "P_FIMPULSE": 65, "P_FACTIVETIME": 66,
    "P_FACTIVE": 67, "P_FAPPLIED": 68,
    "CALLBACK": 1000,
}  # fmt: skip

SIMPLE = {
    "add": "ADD", "sub": "SUB", "mul": "MUL", "div": "DIV", "neg": "NEG", "abs": "ABS", "lt": "LT", "le": "LE",
    "gt": "GT", "ge": "GE", "and": "AND", "or": "OR", "not": "NOT", "norm": "NORM", "dot": "DOT", "cross": "CROSS",
    "sin": "SIN", "cos": "COS", "where": "WHERE", "max_of": "MAXOF", "min_of": "MINOF", "mean_of": "MEANOF",
    "sum_of": "SUMOF", "vec": "VEC", "unit": "UNIT", "clip_norm": "CLIPNORM", "as_unit": "COPY",
    "integral": "INTEGRAL", "running_max": "RUNMAX", "running_min": "RUNMIN", "at_reset": "ATRESET",
    "since_reset": "SINCERESET", "duration_true": "DURTRUE", "time_since": "TIMESINCE", "rising": "RISING",
    "falling": "FALLING", "latch": "LATCH", "once": "ONCE", "sample_on": "SAMPLEON",
}  # fmt: skip
STATS = {"mean": 0, "min": 1, "max": 2, "std": 3, "sum": 4}
KIND = {"real": 0, "bool": 1, "quat": 2}
THING_KIND = {"world": 0, "body": 1, "actor": 2, "anchor": 3}


def _size(shape) -> int:
    return int(np.prod(shape, dtype=int)) if len(shape) else 1


def _supported_shape(shape) -> bool:
    return len(shape) <= 1 and _size(shape) <= 4


def build_program(reg, nodes, used_order, runtime=None):
    """``(Program, native_ids)`` for the nodes the interpreter can run (None when the extension is missing).

    A node is native only if all of its inputs are, so the native region is closed under dependencies and
    needs no Python callbacks; every other node is evaluated in Python from the native values."""
    mod = native.module()
    if mod is None:
        return None, set()
    P = mod.Program()
    P.init(len(nodes))
    P.set_event_count(len(runtime.events) if runtime is not None else 0)
    for b in reg.bodies:
        wp = b.world_pos if b.static and b.world_pos is not None else (0.0, 0.0, 0.0)
        wq = b.world_quat if b.static and b.world_quat is not None else (1.0, 0.0, 0.0, 0.0)
        P.add_body(bool(b.static), int(b.dyn) if not b.static else -1, [float(x) for x in wp],
                   [float(x) for x in wq], [float(x) for x in b.com], float(b.mass))  # fmt: skip
    for a in reg.actors:
        P.add_actor([int(i) for i in a.bodies], float(a.mass))
    path_anchor = set()
    for ai, a in enumerate(reg.anchors):
        rule = getattr(a.rule, "type", "rigid")
        if rule == "path":
            path_anchor.add(ai)
        P.add_anchor(int(a.parent), [float(x) for x in a.pos], [float(x) for x in a.quat], 1 if rule == "translate_only" else 0)

    native_ids: set = set()

    def things_ok(ths) -> bool:
        return all(th.kind != "anchor" or th.index not in path_anchor for th in ths)

    def tcode(th):
        return (THING_KIND[th.kind], int(th.index), th.point == "com")

    def translate(n):
        """(op, params, ints, things) or None for a callback."""
        spec = getattr(n, "spec", None)
        if spec is None or not _supported_shape(n.shape):
            return None
        if any(not _supported_shape(nodes[j].shape) or nodes[j].id not in native_ids for j in n.inputs):
            return None
        shp = [nodes[j].shape for j in n.inputs]

        def broadcastable() -> bool:
            sizes = {_size(s) for s in shp}
            return len(sizes - {1}) <= 1

        if isinstance(spec, ConstNode):
            arr = np.asarray(spec.value, dtype=float).reshape(-1)
            return OPS["CONST"], [float(x) for x in arr], [], []
        if isinstance(spec, TimeNode):
            return OPS["TIME"], [], [], []
        if isinstance(spec, RelNode):
            if spec.measure != "distance":
                return None
            ths = [reg.resolve(t) for t in spec.things]
            if not things_ok(ths):
                return None
            return OPS["REL_DIST"], [], [], [tcode(t) for t in ths]
        if isinstance(spec, PropNode):
            name = spec.prop
            th = reg.resolve(spec.target)
            if name in ("position", "height", "velocity", "speed", "orientation", "angular_velocity", "spin"):
                if th.kind not in ("actor", "body", "anchor") or not things_ok([th]):
                    return None
                op = {"position": "P_POS", "height": "P_HEIGHT", "velocity": "P_VEL", "speed": "P_SPEED",
                      "orientation": "P_ORIENT", "angular_velocity": "P_ANGVEL", "spin": "P_SPIN"}[name]  # fmt: skip
                return OPS[op], [], [], [tcode(th)]
            if runtime is not None and th.kind == "behavior":
                b = runtime.behaviors[th.index]
                if b.machine is None:
                    return None
                if name == "behavior_state":
                    return OPS["P_BSTATE"], [], [th.index], []
                if name == "behavior_time_in_state":
                    return OPS["P_BTIME"], [], [th.index], []
                if name == "behavior_in_state":
                    names = [s.name for s in b.machine.states]
                    return OPS["P_BINSTATE"], [], [th.index, names.index(spec.args["state"])], []
                if name == "behavior_var":
                    return OPS["P_BVAR"], [], [th.index, list(b.machine.variables).index(spec.args["name"])], []
                return None
            if runtime is not None and th.kind == "force":
                op = {"force_impulse": "P_FIMPULSE", "force_active_time": "P_FACTIVETIME", "force_active": "P_FACTIVE",
                      "force_applied": "P_FAPPLIED"}.get(name)  # fmt: skip
                return (OPS[op], [], [th.index], []) if op else None
            if name in ("joint_position", "joint_velocity", "joint_effort") and th.kind == "joint":
                j = reg.joints[th.index]
                if j.type == "ball" or j.dof < 0:
                    return None
                op = {"joint_position": "P_JQ", "joint_velocity": "P_JQD", "joint_effort": "P_JEFF"}[name]
                return OPS[op], [], [int(j.dof)], []
            return None
        if isinstance(spec, OpNode):
            op, a = spec.op, spec.args
            if op == "event" and runtime is not None:
                return OPS["P_EVENT"], [], [runtime.events[str(a["name"])]], []
            if op in SIMPLE:
                if not broadcastable():
                    return None
                return OPS[SIMPLE[op]], [], [], []
            if op == "pow":
                return OPS["POW"], [float(a["n"])], [], []
            if op == "eq":
                return OPS["EQ"], [float(a.get("tol", 0.0))], [], []
            if op == "index":
                return OPS["INDEX"], [], [int(a["i"])], []
            if op == "clip":
                lo, hi = a.get("lo"), a.get("hi")
                return OPS["CLIP"], [0.0 if lo is None else float(lo), 0.0 if hi is None else float(hi)], [lo is not None, hi is not None], []
            if op in ("hold", "cooldown"):
                return OPS[op.upper()], [float(a["seconds"])], [], []
            if op == "derivative":
                wrap = nodes[n.inputs[0]].unit is not None and nodes[n.inputs[0]].unit == RADIAN
                return OPS["DERIV"], [], [int(wrap)], []
            if op == "window":
                return OPS["WINDOW"], [float(a["seconds"])], [STATS[a["stat"]]], []
        return None

    used = set(used_order)
    for n in nodes:
        if n.id not in used:
            continue
        tr = translate(n)
        if tr is None:
            continue
        op, params, ints, ths = tr
        P.set_node(n.id, op, KIND[n.kind], _size(n.shape), len(n.shape), list(n.inputs), params, [int(x) for x in ints], True)
        if ths:
            P.set_things(n.id, ths)
        native_ids.add(n.id)
    P.set_order(sorted(native_ids))
    P.finalize()
    return P, native_ids
