from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from .common import Model, Pose, Vec3


class Rigid(Model):
    type: Literal["rigid"] = "rigid"


class TranslateOnly(Model):
    """Follows the parent point but keeps world orientation (a 'level' frame)."""

    type: Literal["translate_only"] = "translate_only"


class Path(Model):
    """Time-parametrized offset of the anchor in the parent's local frame."""

    type: Literal["path"] = "path"
    kind: Literal["line", "circle", "sine", "keyframes"]
    velocity: Vec3 = (0.0, 0.0, 0.0)  # line
    radius: float = 0.0  # circle
    period: float = 1.0  # circle
    axis: Literal["x", "y", "z"] = "z"  # circle plane normal / sine direction
    amplitude: float = 0.0  # sine
    frequency: float = 1.0  # sine (Hz)
    keyframes: list[tuple[float, Vec3]] = Field(default_factory=list)
    loop: bool = False

    @classmethod
    def line(cls, velocity) -> Path:
        return cls(kind="line", velocity=tuple(velocity))

    @classmethod
    def circle(cls, radius: float, period: float, axis: str = "z") -> Path:
        return cls(kind="circle", radius=radius, period=period, axis=axis)

    @classmethod
    def sine(cls, amplitude: float, frequency: float, axis: str = "z") -> Path:
        return cls(kind="sine", amplitude=amplitude, frequency=frequency, axis=axis)

    @classmethod
    def keyframed(cls, frames, loop: bool = False) -> Path:
        return cls(kind="keyframes", keyframes=[(float(t), tuple(p)) for t, p in frames], loop=loop)


AnchorRule = Annotated[Rigid | TranslateOnly | Path, Field(discriminator="type")]


class Placement(Model):
    """Records how a face anchor was placed (for editing/provenance)."""

    face: str
    body: str
    uv: tuple[float, float] = (0.0, 0.0)
    offset: float = 0.0


class AnchorSpec(Model):
    name: str
    parent: str = "world"  # "world" or a body name local to the actor
    pose: Pose = Field(default_factory=Pose)
    rule: AnchorRule = Field(default_factory=Rigid)
    placement: Placement | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
