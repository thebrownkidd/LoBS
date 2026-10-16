from __future__ import annotations

from pydantic import Field

from .actor import ActorSpec
from .anchor import AnchorSpec
from .common import MaterialSpec, Model, Vec3
from .measure import MeasureNode

SCHEMA_VERSION = 1


class Timing(Model):
    dt: float = 0.002
    substeps: int = 1
    duration: float | None = None


class ContactSettings(Model):
    disabled_group_pairs: list[tuple[int, int]] = Field(default_factory=list)
    default_material: MaterialSpec = Field(
        default_factory=lambda: MaterialSpec(friction=0.8, restitution=0.0, density=1000.0)
    )


class MujocoOptions(Model):
    integrator: str = "implicitfast"
    cone: str = "pyramidal"
    iterations: int = 100
    tolerance: float = 1e-8
    solref_timeconst: float = 0.02


class ChronoOptions(Model):
    system: str = "NSC"
    solver: str = "PSOR"
    max_iterations: int = 100
    collision: str = "BULLET"
    envelope: float = 0.001
    margin: float = 0.001
    threads: int = 1


class BackendOptions(Model):
    mujoco: MujocoOptions = Field(default_factory=MujocoOptions)
    chrono: ChronoOptions = Field(default_factory=ChronoOptions)


class Georeference(Model):
    crs: str | None = None
    offset_m: Vec3 = (0.0, 0.0, 0.0)
    source_up_axis: str = "z"
    source_meters_per_unit: float = 1.0


class ImportRecord(Model):
    source_path: str
    format: str
    source_sha256: str | None = None
    profile_sha256: str | None = None
    overrides_sha256: str | None = None
    motion_mode: str | None = None
    lobs_version: str | None = None
    decisions_path: str | None = None


class ForceSpec(Model):
    """A force law: every tick, a force of (measure-valued) ``magnitude`` acts on ``on`` along a direction
    given by exactly one of ``toward`` (unit vector to another thing), ``direction`` (world vector) or
    ``axis`` ((thing, 'x'|'y'|'z'|'-x'...) frame axis). ``reaction`` adds the equal and opposite force on
    ``toward``. Magnitude is signed: positive attracts toward / acts along the direction."""

    name: str
    on: str
    magnitude: MeasureNode
    toward: str | None = None
    direction: Vec3 | None = None
    axis: tuple[str, str] | None = None
    reaction: bool = True
    tags: list[str] = Field(default_factory=list)


class WorldSpec(Model):
    lobs_schema: int = SCHEMA_VERSION
    name: str = "world"
    description: str = ""
    timing: Timing = Field(default_factory=Timing)
    gravity: Vec3 = (0.0, 0.0, -9.81)
    contact: ContactSettings = Field(default_factory=ContactSettings)
    actors: list[ActorSpec] = Field(default_factory=list)
    anchors: list[AnchorSpec] = Field(default_factory=list)  # world-fixed
    forces: list[ForceSpec] = Field(default_factory=list)
    measures: dict[str, MeasureNode] = Field(default_factory=dict)
    backend_options: BackendOptions = Field(default_factory=BackendOptions)
    georeference: Georeference | None = None
    imports: list[ImportRecord] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
