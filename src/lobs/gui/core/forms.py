"""Schema-driven forms: describe the editable fields of any spec model (no Qt).

Widgets build one editor per ``FieldSpec``; new schema fields appear automatically."""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass, field

from pydantic import BaseModel


@dataclass
class FieldSpec:
    path: tuple  # attribute names from the edited object
    label: str
    kind: str  # float int bool str vec2 vec3 vec4 choice measure union json
    value: object = None
    choices: list = field(default_factory=list)
    optional: bool = False
    description: str = ""

    @property
    def key(self) -> str:
        return ".".join(self.path)


def _unwrap(ann):
    """(inner annotation, optional)."""
    origin = typing.get_origin(ann)
    if origin is typing.Annotated:
        return _unwrap(typing.get_args(ann)[0])
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) != len(typing.get_args(ann)) and len(args) == 1:
            inner, _ = _unwrap(args[0])
            return inner, True
        return ann, len(args) != len(typing.get_args(ann))
    return ann, False


def _is_measure(ann) -> bool:
    return "PropNode" in str(ann) and "OpNode" in str(ann)


def _kind(ann):
    ann, optional = _unwrap(ann)
    origin, args = typing.get_origin(ann), typing.get_args(ann)
    if _is_measure(ann):
        return "measure", [], optional
    if ann is bool:
        return "bool", [], optional
    if ann is int:
        return "int", [], optional
    if ann is float:
        return "float", [], optional
    if ann is str:
        return "str", [], optional
    if origin is typing.Literal:
        return "choice", list(args), optional
    if origin is tuple and args and all(a is float for a in args) and len(args) in (2, 3, 4):
        return {2: "vec2", 3: "vec3", 4: "vec4"}[len(args)], [], optional
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return "model", [], optional
    if origin in (typing.Union, types.UnionType) and all(
        isinstance(a, type) and issubclass(a, BaseModel) for a in args
    ):
        return "union", [a.__name__ for a in args], optional
    return "json", [], optional


def fields_of(obj, prefix: tuple = (), skip: tuple = ()) -> list[FieldSpec]:
    """Editable fields of a pydantic model instance, recursing into nested models (dotted paths)."""
    out: list[FieldSpec] = []
    for name, info in type(obj).model_fields.items():
        if name in skip:
            continue
        value = getattr(obj, name)
        kind, choices, optional = _kind(info.annotation)
        path = (*prefix, name)
        label = ".".join(path)
        desc = info.description or ""
        if kind == "model" and value is not None:
            out.extend(fields_of(value, path, skip))
        elif kind == "union" and value is not None:
            out.append(
                FieldSpec(path, label, "union", type(value).__name__, choices, optional, desc)
            )
            out.extend(fields_of(value, path, skip))
        elif kind in ("model", "union") and value is None:
            out.append(FieldSpec(path, label, "json", None, [], True, desc))
        else:
            out.append(FieldSpec(path, label, kind, value, choices, optional, desc))
    return out


def get_path(obj, path):
    for p in path:
        obj = getattr(obj, p)
    return obj


def set_path(obj, path, value) -> None:
    """Assign a leaf by path; the models validate on assignment, so bad values raise."""
    parent = get_path(obj, path[:-1])
    setattr(parent, path[-1], value)
