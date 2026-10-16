from __future__ import annotations

from typing import Literal

from pydantic import Field

from .anchor import AnchorSpec
from .common import MaterialSpec, Model, Pose, Shape, Vec3


class GeomSpec(Model):
    shape: Shape
    pose: Pose = Field(default_factory=Pose)
    role: Literal["both", "visual", "collision"] = "both"
    material: MaterialSpec | None = None
    rgba: tuple[float, float, float, float] | None = None
    source_id: str | None = None


class InertialSpec(Model):
    mode: Literal["auto", "explicit"] = "auto"
    mass: float | None = None
    density: float | None = None
    com: Vec3 = (0.0, 0.0, 0.0)
    inertia: tuple[Vec3, Vec3, Vec3] | None = None  # about COM, body axes (explicit mode)


class SourceRef(Model):
    format: str | None = None
    source_id: str | None = None
    source_path: str | None = None
    semantic_class: str | None = None


class BodySpec(Model):
    name: str
    parent: str | None = None
    pose: Pose = Field(default_factory=Pose)  # relative to parent body (root: relative to actor)
    geoms: list[GeomSpec] = Field(default_factory=list)
    inertial: InertialSpec = Field(default_factory=InertialSpec)
    collision_group: int = Field(default=0, ge=0, le=15)
    tags: list[str] = Field(default_factory=list)
    source: SourceRef | None = None


class JointSpec(Model):
    name: str
    type: Literal["hinge", "slide", "fixed", "ball"] = "hinge"
    parent: str | None = None
    child: str | None = None
    pos: Vec3 = (0.0, 0.0, 0.0)  # in child body frame
    axis: Vec3 = (0.0, 0.0, 1.0)  # in child body frame
    range: tuple[float, float] | None = None
    damping: float = 0.0
    armature: float = 0.0


class ActuatorSpec(Model):
    name: str
    joint: str
    kind: Literal["effort", "position", "velocity"] = "effort"
    kp: float = 0.0
    kd: float = 0.0
    ctrl_range: tuple[float, float] | None = None
    effort_limit: float | None = None


class InitialState(Model):
    pose: Pose | None = None
    linvel: Vec3 = (0.0, 0.0, 0.0)  # world frame, velocity of the root reference origin
    angvel: Vec3 = (0.0, 0.0, 0.0)  # world frame
    # joint name -> position (hinge: rad, slide: m, ball: rotation vector (3,))
    q: dict[str, float | tuple[float, float, float]] = Field(default_factory=dict)
    qd: dict[str, float | tuple[float, float, float]] = Field(default_factory=dict)


class ActorSpec(Model):
    name: str
    root: Literal["free", "fixed"] = "free"
    pose: Pose = Field(default_factory=Pose)
    bodies: list[BodySpec] = Field(default_factory=list)
    joints: list[JointSpec] = Field(default_factory=list)
    actuators: list[ActuatorSpec] = Field(default_factory=list)
    anchors: list[AnchorSpec] = Field(default_factory=list)
    initial: InitialState = Field(default_factory=InitialState)
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    source: SourceRef | None = None
