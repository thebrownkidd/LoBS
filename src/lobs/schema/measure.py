from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .common import Model


class PropNode(Model):
    kind: Literal["prop"] = "prop"
    prop: str
    target: str | None = None
    args: dict = Field(default_factory=dict)


class RelNode(Model):
    kind: Literal["rel"] = "rel"
    measure: Literal["distance", "angle"]
    things: list[str]
    angle_kind: str | None = None
    axis: str | tuple[float, float, float] = "z"
    axis_b: str | tuple[float, float, float] | None = None
    frame: str | None = None
    signed_about: str | tuple[float, float, float] | None = None


class ConstNode(Model):
    kind: Literal["const"] = "const"
    value: float | bool | list[float]
    unit: str | None = None  # None = unitless literal that adopts its partner's unit


class TimeNode(Model):
    kind: Literal["time"] = "time"


class RefNode(Model):
    kind: Literal["ref"] = "ref"
    name: str


class OpNode(Model):
    kind: Literal["op"] = "op"
    op: str
    inputs: list[MeasureNode] = Field(default_factory=list)
    args: dict = Field(default_factory=dict)


class CustomNode(Model):
    kind: Literal["custom"] = "custom"
    fn: str
    inputs: dict[str, MeasureNode] = Field(default_factory=dict)
    unit: str | None = None
    shape: list[int] = Field(default_factory=list)


MeasureNode = Annotated[
    PropNode | RelNode | ConstNode | TimeNode | RefNode | OpNode | CustomNode,
    Field(discriminator="kind"),
]
OpNode.model_rebuild()
CustomNode.model_rebuild()
