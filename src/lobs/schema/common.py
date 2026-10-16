from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

Vec3 = tuple[float, float, float]
Quat4 = tuple[float, float, float, float]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Pose(Model):
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat4 = (1.0, 0.0, 0.0, 0.0)

    @field_validator("quat")
    @classmethod
    def _unit(cls, q):
        a = np.asarray(q, dtype=float)
        n = np.linalg.norm(a)
        if n < 1e-12:
            raise ValueError("zero quaternion")
        return tuple(float(x) for x in a / n)


class MaterialSpec(Model):
    name: str | None = None
    friction: float | None = None
    restitution: float | None = None
    density: float | None = None


# ---- shapes (local z axis conventions; see plan section 4) ----
class Box(Model):
    type: Literal["box"] = "box"
    size: Vec3  # full extents


class Sphere(Model):
    type: Literal["sphere"] = "sphere"
    radius: float


class Cylinder(Model):
    type: Literal["cylinder"] = "cylinder"
    radius: float
    length: float  # full height


class Capsule(Model):
    type: Literal["capsule"] = "capsule"
    radius: float
    length: float  # cylindrical segment only


class Plane(Model):
    type: Literal["plane"] = "plane"


class ConvexMesh(Model):
    type: Literal["convex_mesh"] = "convex_mesh"
    file: str


class ConvexSet(Model):
    type: Literal["convex_set"] = "convex_set"
    files: list[str]


class TriangleMesh(Model):
    type: Literal["triangle_mesh"] = "triangle_mesh"
    file: str


class Heightfield(Model):
    type: Literal["heightfield"] = "heightfield"
    file: str
    size: tuple[float, float]


Shape = Annotated[
    Box | Sphere | Cylinder | Capsule | Plane | ConvexMesh | ConvexSet | TriangleMesh | Heightfield,
    Field(discriminator="type"),
]
