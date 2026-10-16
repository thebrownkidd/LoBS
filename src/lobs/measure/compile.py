"""Compile a MeasureNode tree into a flat, typed, unit-checked evaluation list (plan 8.6)."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .. import math3d as m3
from ..errors import MeasureError
from ..registry import Registry
from ..schema.measure import (
    ConstNode,
    CustomNode,
    OpNode,
    PropNode,
    RefNode,
    RelNode,
    TimeNode,
)
from . import catalog
from .context import PERTURB
from .functions import FUNCTIONS
from .units import DIMENSIONLESS, RADIAN, SECOND, compatible, fmt, parse

_PROV_ORDER = {"model": 0, "engine": 1, "computed": 2, "estimated": 3}
RATE_OK_NEEDS = frozenset({"pose", "joints", "mass"})


@dataclass
class CNode:
    id: int
    kind: str  # real | bool | quat
    shape: tuple
    unit: object  # pint Unit, or None for a unitless literal
    needs: frozenset
    prov: str
    caps: frozenset
    inputs: tuple
    fn: Callable
    stateful: bool = False
    state: dict = field(default_factory=dict)
    desc: str = ""


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(a, b)


def _worst(*provs: str) -> str:
    return max(provs, key=lambda p: _PROV_ORDER[p])


class Compiler:
    def __init__(self, reg: Registry, named: dict):
        self.reg = reg
        self.named = named
        self.nodes: list[CNode] = []
        self._memo: dict = {}
        self._named_ids: dict = {}
        self._stack: list = []

    # ------------------------------------------------------------ entry points
    def compile_named(self, name: str) -> int:
        if name in self._named_ids:
            return self._named_ids[name]
        if name not in self.named:
            raise MeasureError(f"unknown measure '{name}'")
        if name in self._stack:
            raise MeasureError("measure reference cycle: " + " -> ".join([*self._stack, name]))
        self._stack.append(name)
        try:
            nid = self.compile(self.named[name])
        except MeasureError as e:
            raise MeasureError(f"measure '{name}': {e}") from None
        finally:
            self._stack.pop()
        self._named_ids[name] = nid
        return nid

    def compile(self, node) -> int:
        if isinstance(node, RefNode):
            return self.compile_named(node.name)
        key = node.model_dump_json()
        if key in self._memo:
            return self._memo[key]
        handler = {
            PropNode: self._prop, RelNode: self._rel, ConstNode: self._const,
            TimeNode: self._time, OpNode: self._op, CustomNode: self._custom,
        }[type(node)]  # fmt: skip
        nid = handler(node)
        self.nodes[nid].desc = key
        self._memo[key] = nid
        return nid

    def _add(self, kind, shape, unit, needs, prov, caps, inputs, fn, stateful=False) -> int:
        nid = len(self.nodes)
        self.nodes.append(
            CNode(
                nid,
                kind,
                tuple(shape),
                unit,
                frozenset(needs),
                prov,
                frozenset(caps),
                tuple(inputs),
                fn,
                stateful,
            )  # fmt: skip
        )
        return nid

    def _inherit(self, ids):
        ns = [self.nodes[i] for i in ids]
        needs = frozenset().union(*(n.needs for n in ns)) if ns else frozenset()
        caps = frozenset().union(*(n.caps for n in ns)) if ns else frozenset()
        prov = _worst("model", *(n.prov for n in ns))
        return needs, prov, caps

    # ------------------------------------------------------------ leaves
    def _resolve(self, path, allowed, what):
        try:
            th = self.reg.resolve(path)
        except Exception as e:
            raise MeasureError(str(e)) from None
        if th.kind not in allowed:
            raise MeasureError(f"{what}: '{path}' is a {th.kind}; expected {'/'.join(allowed)}")
        return th

    def _prop(self, node: PropNode) -> int:
        d = catalog.PROPS.get(node.prop)
        if d is None:
            raise MeasureError(
                f"unknown property '{node.prop}' (known: {', '.join(sorted(catalog.PROPS))})"
            )
        if node.target is None:
            raise MeasureError(f"property '{node.prop}' needs a target")
        th = self._resolve(node.target, d.targets, f"property '{node.prop}'")
        args = dict(node.args)
        unknown = set(args) - set(d.arg_names)
        if unknown:
            raise MeasureError(f"property '{node.prop}' got unknown arguments {sorted(unknown)}")
        if "with" in args:
            other = args.pop("with")
            args["_with"] = self._resolve(other, ("actor", "body"), "'with_'") if other else None
        if (
            th.kind == "joint"
            and node.prop in ("joint_effort", "joint_limit_margin")
            and self.reg.joints[th.index].type == "ball"
        ):
            raise MeasureError(f"{node.prop} is not defined for ball joint '{th.path}'")
        if th.kind == "joint" and self.reg.joints[th.index].dof < 0:
            raise MeasureError(f"joint '{th.path}' is fixed and has no state")
        unit = d.unit(self.reg, th, args) if callable(d.unit) else d.unit
        shape = d.shape(self.reg, th, args) if callable(d.shape) else d.shape
        fn = d.fn

        def run(ctx, ins, state, th=th, args=args, fn=fn):
            return np.asarray(fn(ctx, th, args, state), dtype=float if d.kind != "bool" else bool)

        return self._add(d.kind, shape, parse(unit), d.needs, d.prov,
                         {d.cap} if d.cap else set(), (), run, d.stateful)  # fmt: skip

    def _rel(self, node: RelNode) -> int:
        n = len(node.things)
        if node.measure == "distance":
            if n != 2:
                raise MeasureError("distance needs exactly 2 things")
            things = [
                self._resolve(t, ("actor", "body", "anchor", "world"), "rel") for t in node.things
            ]
            return self._add("real", (), parse("meter"), {"pose"}, "computed", set(), (),
                             lambda ctx, ins, st, th=things: np.asarray(catalog.eval_distance(ctx, th)))  # fmt: skip
        things = [
            self._resolve(t, ("actor", "body", "anchor", "world"), "rel") for t in node.things
        ]
        frame_thing = None
        if node.frame not in (None, "a", "b", "v", "world"):
            frame_thing = self._resolve(node.frame, ("actor", "body", "anchor", "world"), "frame")
        if n == 2:
            if node.angle_kind not in ("polar", "elevation", "azimuth", "axes", "rotation"):
                raise MeasureError(
                    "angle between two things needs a kind: polar, elevation, azimuth, axes or rotation"
                )
            if node.signed_about is not None:
                raise MeasureError("signed_about only applies to the three-thing angle")
            if node.angle_kind in ("polar", "elevation", "azimuth") and node.axis_b is not None:
                raise MeasureError("axis_b only applies to angle kind 'axes'")
            catalog.parse_axis(node.axis)
            if node.axis_b is not None:
                catalog.parse_axis(node.axis_b)
            fn = lambda ctx, ins, st, th=things, nd=node, ft=frame_thing: np.asarray(  # noqa: E731
                catalog.eval_angle_pair(ctx, th, nd, ft)
            )
        elif n == 3:
            if node.angle_kind is not None:
                raise MeasureError(
                    "the three-thing angle is the interior angle at the middle thing"
                )
            if node.signed_about is not None:
                catalog.parse_axis(node.signed_about)
            fn = lambda ctx, ins, st, th=things, nd=node, ft=frame_thing: np.asarray(  # noqa: E731
                catalog.eval_angle_triple(ctx, th, nd, ft)
            )
        else:
            raise MeasureError("angle needs 2 or 3 things")
        return self._add("real", (), RADIAN, {"pose"}, "computed", set(), (), fn)

    def _const(self, node: ConstNode) -> int:
        v = node.value
        kind = "bool" if isinstance(v, bool) else "real"
        arr = np.asarray(v, dtype=bool if kind == "bool" else float)
        return self._add(kind, arr.shape, parse(node.unit), set(), "model", set(), (),
                         lambda ctx, ins, st, a=arr: a)  # fmt: skip

    def _time(self, node) -> int:
        return self._add("real", (), SECOND, set(), "model", set(), (),
                         lambda ctx, ins, st: np.asarray(ctx.t))  # fmt: skip

    def _custom(self, node: CustomNode) -> int:
        fn = FUNCTIONS.get(node.fn)
        if fn is None:
            raise MeasureError(
                f"custom function '{node.fn}' is not registered (lobs.register_function)"
            )
        names = list(node.inputs)
        ids = [self.compile(node.inputs[k]) for k in names]
        needs, prov, caps = self._inherit(ids)

        def run(ctx, ins, st, fn=fn, names=names):
            return np.asarray(fn(**dict(zip(names, ins, strict=True))), dtype=float)

        return self._add("real", tuple(node.shape), parse(node.unit), needs, _worst(prov, "computed"),
                         caps, ids, run)  # fmt: skip

    # ------------------------------------------------------------ operators
    def _op(self, node: OpNode) -> int:
        ids = [self.compile(i) for i in node.inputs]
        ins = [self.nodes[i] for i in ids]
        needs, prov, caps = self._inherit(ids)
        op = node.op
        h = getattr(self, f"_op_{op}", None)
        if h is None:
            raise MeasureError(f"unknown operator '{op}'")
        return h(node, ids, ins, needs, prov, caps)

    # -- helpers
    def _need_real(self, op, ins):
        for n in ins:
            if n.kind != "real":
                raise MeasureError(f"'{op}' needs numeric inputs, got {n.kind}")

    def _need_n(self, op, ins, k):
        if len(ins) != k:
            raise MeasureError(f"'{op}' needs {k} inputs, got {len(ins)}")

    def _bshape(self, op, ins):
        try:
            return np.broadcast_shapes(*(n.shape for n in ins))
        except ValueError:
            raise MeasureError(f"'{op}': incompatible shapes {[n.shape for n in ins]}") from None

    def _match_units(self, op, a, b):
        if a.unit is None:
            return b.unit
        if b.unit is None or compatible(a.unit, b.unit):
            return a.unit
        raise MeasureError(f"'{op}': unit mismatch ({fmt(a.unit)} vs {fmt(b.unit)})")

    def _binary(self, node, ids, ins, needs, prov, caps, kind, unit, fn, shape=None):
        shape = self._bshape(node.op, ins) if shape is None else shape
        return self._add(kind, shape, unit, needs, _worst(prov, "model"), caps, ids,
                         lambda ctx, v, st: np.asarray(fn(v[0], v[1])))  # fmt: skip

    def _op_add(self, node, ids, ins, needs, prov, caps):
        self._need_n("add", ins, 2)
        self._need_real("add", ins)  # noqa: E702
        return self._binary(
            node, ids, ins, needs, prov, caps, "real", self._match_units("add", *ins), np.add
        )

    def _op_sub(self, node, ids, ins, needs, prov, caps):
        self._need_n("sub", ins, 2)
        self._need_real("sub", ins)  # noqa: E702
        return self._binary(
            node, ids, ins, needs, prov, caps, "real", self._match_units("sub", *ins), np.subtract
        )

    def _mul_unit(self, a, b, div=False):
        if a.unit is None and b.unit is None:
            return None
        ua, ub = a.unit or DIMENSIONLESS, b.unit or DIMENSIONLESS
        return ua / ub if div else ua * ub

    def _op_mul(self, node, ids, ins, needs, prov, caps):
        self._need_n("mul", ins, 2)
        self._need_real("mul", ins)  # noqa: E702
        return self._binary(
            node, ids, ins, needs, prov, caps, "real", self._mul_unit(*ins), np.multiply
        )

    def _op_div(self, node, ids, ins, needs, prov, caps):
        self._need_n("div", ins, 2)
        self._need_real("div", ins)  # noqa: E702
        return self._binary(
            node, ids, ins, needs, prov, caps, "real", self._mul_unit(*ins, div=True), _safe_div
        )

    def _op_pow(self, node, ids, ins, needs, prov, caps):
        self._need_n("pow", ins, 1)
        self._need_real("pow", ins)  # noqa: E702
        n = float(node.args["n"])
        unit = None if ins[0].unit is None else ins[0].unit ** n
        return self._add("real", ins[0].shape, unit, needs, prov, caps, ids,
                         lambda ctx, v, st: np.power(v[0], n))  # fmt: skip

    def _unary(self, node, ids, ins, needs, prov, caps, fn, unit=..., shape=None, kind="real"):
        self._need_n(node.op, ins, 1)
        return self._add(kind, ins[0].shape if shape is None else shape,
                         ins[0].unit if unit is ... else unit, needs, prov, caps, ids,
                         lambda ctx, v, st: np.asarray(fn(v[0])))  # fmt: skip

    def _op_neg(self, node, ids, ins, needs, prov, caps):
        self._need_real("neg", ins)
        return self._unary(node, ids, ins, needs, prov, caps, np.negative)

    def _op_abs(self, node, ids, ins, needs, prov, caps):
        self._need_real("abs", ins)
        return self._unary(node, ids, ins, needs, prov, caps, np.abs)

    def _cmp(self, node, ids, ins, needs, prov, caps, fn):
        self._need_n(node.op, ins, 2)
        self._need_real(node.op, ins)  # noqa: E702
        self._match_units(node.op, *ins)
        return self._binary(node, ids, ins, needs, prov, caps, "bool", DIMENSIONLESS, fn)

    def _op_lt(self, *a):
        return self._cmp(*a, np.less)  # noqa: E704

    def _op_le(self, *a):
        return self._cmp(*a, np.less_equal)  # noqa: E704

    def _op_gt(self, *a):
        return self._cmp(*a, np.greater)  # noqa: E704

    def _op_ge(self, *a):
        return self._cmp(*a, np.greater_equal)  # noqa: E704

    def _op_eq(self, node, ids, ins, needs, prov, caps):
        tol = float(node.args.get("tol", 0.0))
        return self._cmp(node, ids, ins, needs, prov, caps, lambda a, b: np.abs(a - b) <= tol)

    def _logic(self, node, ids, ins, needs, prov, caps, fn):
        self._need_n(node.op, ins, 2)
        if any(n.kind != "bool" for n in ins):
            raise MeasureError(f"'{node.op}' needs boolean inputs")
        return self._binary(node, ids, ins, needs, prov, caps, "bool", DIMENSIONLESS, fn)

    def _op_and(self, *a):
        return self._logic(*a, np.logical_and)  # noqa: E704

    def _op_or(self, *a):
        return self._logic(*a, np.logical_or)  # noqa: E704

    def _op_not(self, node, ids, ins, needs, prov, caps):
        self._need_n("not", ins, 1)
        if ins[0].kind != "bool":
            raise MeasureError("'not' needs a boolean input")
        return self._unary(
            node, ids, ins, needs, prov, caps, np.logical_not, DIMENSIONLESS, kind="bool"
        )

    def _op_index(self, node, ids, ins, needs, prov, caps):
        self._need_n("index", ins, 1)
        self._need_real("index", ins)  # noqa: E702
        i = node.args["i"]
        if not ins[0].shape or not (-ins[0].shape[-1] <= i < ins[0].shape[-1]):
            raise MeasureError(f"index {i} out of range for shape {ins[0].shape}")
        return self._unary(
            node, ids, ins, needs, prov, caps, lambda a: a[..., i], shape=ins[0].shape[:-1]
        )

    def _op_norm(self, node, ids, ins, needs, prov, caps):
        self._need_n("norm", ins, 1)
        self._need_real("norm", ins)  # noqa: E702
        if not ins[0].shape:
            raise MeasureError("norm needs a vector")
        return self._unary(node, ids, ins, needs, prov, caps, lambda a: np.linalg.norm(a, axis=-1),
                           shape=ins[0].shape[:-1])  # fmt: skip

    def _op_dot(self, node, ids, ins, needs, prov, caps):
        self._need_n("dot", ins, 2)
        self._need_real("dot", ins)  # noqa: E702
        if any(n.shape != (3,) for n in ins):
            raise MeasureError("dot needs two 3-vectors")
        return self._binary(
            node, ids, ins, needs, prov, caps, "real", self._mul_unit(*ins), np.dot, ()
        )

    def _op_cross(self, node, ids, ins, needs, prov, caps):
        self._need_n("cross", ins, 2)
        self._need_real("cross", ins)  # noqa: E702
        if any(n.shape != (3,) for n in ins):
            raise MeasureError("cross needs two 3-vectors")
        return self._binary(
            node, ids, ins, needs, prov, caps, "real", self._mul_unit(*ins), np.cross, (3,)
        )

    def _trig(self, node, ids, ins, needs, prov, caps, fn):
        self._need_n(node.op, ins, 1)
        self._need_real(node.op, ins)  # noqa: E702
        if ins[0].unit is not None and not compatible(ins[0].unit, RADIAN):
            raise MeasureError(
                f"'{node.op}' needs an angle or dimensionless input, got {fmt(ins[0].unit)}"
            )
        return self._unary(node, ids, ins, needs, prov, caps, fn, DIMENSIONLESS)

    def _op_sin(self, *a):
        return self._trig(*a, np.sin)  # noqa: E704

    def _op_cos(self, *a):
        return self._trig(*a, np.cos)  # noqa: E704

    def _op_clip(self, node, ids, ins, needs, prov, caps):
        self._need_real("clip", ins)
        lo, hi = node.args.get("lo"), node.args.get("hi")
        return self._unary(node, ids, ins, needs, prov, caps, lambda a: np.clip(a, lo, hi))

    def _op_where(self, node, ids, ins, needs, prov, caps):
        self._need_n("where", ins, 3)
        if ins[0].kind != "bool":
            raise MeasureError("'where' needs a boolean condition")
        self._need_real("where", ins[1:])
        unit = self._match_units("where", ins[1], ins[2])
        return self._add("real", self._bshape("where", ins), unit, needs, prov, caps, ids,
                         lambda ctx, v, st: np.where(v[0], v[1], v[2]))  # fmt: skip

    def _reduce_n(self, node, ids, ins, needs, prov, caps, fn):
        if not ins:
            raise MeasureError(f"'{node.op}' needs at least one input")
        self._need_real(node.op, ins)
        unit = None
        for n in ins:
            if n.unit is None:
                continue
            if unit is None:
                unit = n.unit
            elif not compatible(unit, n.unit):
                raise MeasureError(f"'{node.op}': unit mismatch ({fmt(unit)} vs {fmt(n.unit)})")
        return self._add("real", self._bshape(node.op, ins), unit, needs, prov, caps, ids,
                         lambda ctx, v, st: fn(np.broadcast_arrays(*v), axis=0))  # fmt: skip

    def _op_max_of(self, *a):
        return self._reduce_n(*a, np.max)  # noqa: E704

    def _op_min_of(self, *a):
        return self._reduce_n(*a, np.min)  # noqa: E704

    def _op_mean_of(self, *a):
        return self._reduce_n(*a, np.mean)  # noqa: E704

    def _op_sum_of(self, *a):
        return self._reduce_n(*a, np.sum)  # noqa: E704

    # -- time-dependent
    def _closure(self, root: int) -> list[CNode]:
        seen, stack = set(), [root]
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            stack.extend(self.nodes[i].inputs)
        return [self.nodes[i] for i in sorted(seen)]

    def _op_rate(self, node, ids, ins, needs, prov, caps):
        self._need_n("rate", ins, 1)
        x = ins[0]
        if x.kind != "real":
            raise MeasureError(f"rate: cannot differentiate a {x.kind} value")
        closure = self._closure(ids[0])
        for n in closure:
            if n.stateful:
                raise MeasureError(
                    "rate: input contains stateful (time-dependent) measures; use .derivative()"
                )
            bad = n.needs - RATE_OK_NEEDS
            if bad:
                raise MeasureError(
                    f"rate: input depends on {sorted(bad)} which is not a pose-like quantity; "
                    "use .derivative() for an estimated rate"
                )
        wrap = x.unit is not None and x.unit == RADIAN
        root = ids[0]

        def run(ctx, v, st, closure=closure, root=root, wrap=wrap):
            def at(delta):
                c = ctx.advanced(delta)
                vals = {}
                for n in closure:
                    vals[n.id] = n.fn(c, [vals[i] for i in n.inputs], n.state)
                return vals[root]

            d = at(PERTURB) - at(-PERTURB)
            if wrap:
                d = m3.wrap_angle(d)
            return d / (2 * PERTURB)

        unit = None if x.unit is None else x.unit / SECOND
        return self._add("real", x.shape, unit, needs, _worst(prov, "computed"), caps, ids, run)

    def _stateful1(
        self, node, ids, ins, needs, prov, caps, run, unit=..., kind="real", require=None
    ):
        self._need_n(node.op, ins, 1)
        if require and ins[0].kind != require:
            raise MeasureError(f"'{node.op}' needs a {require} input")
        if require is None:
            self._need_real(node.op, ins)
        return self._add(kind, ins[0].shape if kind != "bool" else (), ins[0].unit if unit is ... else unit,
                         needs, prov, caps, ids, run, True)  # fmt: skip

    def _op_derivative(self, node, ids, ins, needs, prov, caps):
        wrap = ins[0].unit is not None and ins[0].unit == RADIAN

        def run(ctx, v, st):
            x = np.asarray(v[0], dtype=float)
            if "t" not in st:
                out = np.full(x.shape, np.nan)
            else:
                d = x - st["x"]
                if wrap:
                    d = m3.wrap_angle(d)
                dt = ctx.t - st["t"]
                out = d / dt if dt > 0 else np.full(x.shape, np.nan)
            st["t"], st["x"] = ctx.t, x
            return out

        unit = None if ins[0].unit is None else ins[0].unit / SECOND
        return self._stateful1(node, ids, ins, needs, _worst(prov, "estimated"), caps, run, unit)

    def _op_integral(self, node, ids, ins, needs, prov, caps):
        def run(ctx, v, st):
            x = np.asarray(v[0], dtype=float)
            if "t" not in st:
                st["acc"] = np.zeros_like(x)
            else:
                st["acc"] = st["acc"] + 0.5 * (x + st["x"]) * (ctx.t - st["t"])
            st["t"], st["x"] = ctx.t, x
            return st["acc"]

        unit = None if ins[0].unit is None else ins[0].unit * SECOND
        return self._stateful1(node, ids, ins, needs, _worst(prov, "estimated"), caps, run, unit)

    def _op_running_max(self, node, ids, ins, needs, prov, caps):
        def run(ctx, v, st):
            st["m"] = np.fmax(st["m"], v[0]) if "m" in st else np.asarray(v[0], dtype=float)
            return st["m"]

        return self._stateful1(node, ids, ins, needs, prov, caps, run)

    def _op_running_min(self, node, ids, ins, needs, prov, caps):
        def run(ctx, v, st):
            st["m"] = np.fmin(st["m"], v[0]) if "m" in st else np.asarray(v[0], dtype=float)
            return st["m"]

        return self._stateful1(node, ids, ins, needs, prov, caps, run)

    def _op_at_reset(self, node, ids, ins, needs, prov, caps):
        def run(ctx, v, st):
            st.setdefault("x0", np.asarray(v[0], dtype=float).copy())
            return st["x0"]

        return self._stateful1(node, ids, ins, needs, prov, caps, run)

    def _op_since_reset(self, node, ids, ins, needs, prov, caps):
        def run(ctx, v, st):
            st.setdefault("x0", np.asarray(v[0], dtype=float).copy())
            return np.asarray(v[0], dtype=float) - st["x0"]

        return self._stateful1(node, ids, ins, needs, prov, caps, run)

    def _op_window(self, node, ids, ins, needs, prov, caps):
        seconds, stat = float(node.args["seconds"]), node.args["stat"]
        if seconds <= 0 or stat not in ("mean", "min", "max", "std", "sum"):
            raise MeasureError("window needs seconds > 0 and stat in mean/min/max/std/sum")
        fn = {"mean": np.mean, "min": np.min, "max": np.max, "std": np.std, "sum": np.sum}[stat]

        def run(ctx, v, st):
            buf = st.setdefault("buf", deque())
            buf.append((ctx.t, np.asarray(v[0], dtype=float)))
            while buf[0][0] < ctx.t - seconds - 1e-12:
                buf.popleft()
            return fn(np.stack([b[1] for b in buf]), axis=0)

        unit = ins[0].unit
        if stat == "sum" and unit is None:
            unit = None
        return self._stateful1(node, ids, ins, needs, prov, caps, run, unit)

    def _op_duration_true(self, node, ids, ins, needs, prov, caps):
        def run(ctx, v, st):
            if bool(v[0]):
                st.setdefault("t0", ctx.t)
                return np.asarray(ctx.t - st["t0"])
            st.pop("t0", None)
            return np.asarray(0.0)

        return self._stateful1(
            node, ids, ins, needs, prov, caps, run, SECOND, require="bool", kind="real"
        )

    def _op_time_since(self, node, ids, ins, needs, prov, caps):
        def run(ctx, v, st):
            on = bool(v[0])
            if on and not st.get("prev", False):
                st["edge"] = ctx.t
            st["prev"] = on
            return np.asarray(ctx.t - st["edge"] if "edge" in st else np.nan)

        return self._stateful1(
            node, ids, ins, needs, prov, caps, run, SECOND, require="bool", kind="real"
        )
