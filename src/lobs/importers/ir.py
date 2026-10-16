"""Scene IR: what readers produce. Readers translate format -> IR and make NO physics decisions
(other than copying physics data that is explicitly present in the source)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IRMaterial:
    name: str | None = None
    density: float | None = None
    friction: float | None = None
    restitution: float | None = None
    color: tuple | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class IRPhysics:
    """Physics data explicitly present in the source (None = not specified)."""

    rigid: bool | None = None  # False -> fixed in the world
    mass: float | None = None
    com: tuple | None = None  # in node frame
    inertia: np.ndarray | None = None  # 3x3 about COM, node axes
    collision_enabled: bool | None = None
    collision_hint: str | None = None  # convex_hull | decompose | mesh | box_fit | none


@dataclass
class IRGeom:
    kind: str  # mesh | box | sphere | cylinder | capsule | plane
    vertices: np.ndarray | None = None  # (n, 3) in node frame (mesh)
    faces: np.ndarray | None = None  # (m, 3)
    size: tuple | None = None  # box full extents
    radius: float | None = None
    length: float | None = None
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))  # geom frame in node frame
    role: str = "both"  # both | visual | collision
    material: IRMaterial | None = None


@dataclass
class IRNode:
    id: str  # stable source identity (GlobalId, prim path, node path ...)
    name: str
    parent: str | None = None
    local: np.ndarray = field(
        default_factory=lambda: np.eye(4)
    )  # relative to parent (roots: world)
    semantic_class: str | None = None
    tags: list = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    geoms: list = field(default_factory=list)
    material: IRMaterial | None = None
    physics: IRPhysics | None = None
    sites: list = field(
        default_factory=list
    )  # [(name, pos(3), quat(4))] in the node frame -> anchors


@dataclass
class IRJoint:
    id: str
    name: str
    type: str  # revolute | continuous | prismatic | fixed | spherical | floating | planar
    parent_node: str
    child_node: str
    pos: tuple = (0.0, 0.0, 0.0)  # in child node frame
    axis: tuple = (0.0, 0.0, 1.0)  # in child node frame
    limits: tuple | None = None
    damping: float = 0.0
    effort_limit: float | None = None
    drive: tuple | None = None  # (kp, kd)
    armature: float = 0.0
    actuator: dict | None = (
        None  # {'kind': effort|position|velocity, 'kp', 'kd', 'ctrl_range', 'effort_limit'}
    )


@dataclass
class SourceInfo:
    path: str
    format: str
    sha256: str | None = None
    meters_per_unit: float | None = None  # None = unknown
    up_axis: str | None = None  # 'y' | 'z' | None (unknown)
    georef: dict | None = None  # {'crs': str|None, 'offset': (x, y, z)} in source units
    reader_version: str = "1"
    notes: list = field(default_factory=list)


@dataclass
class IRWorldPhysics:
    gravity: tuple | None = None
    timestep: float | None = None


@dataclass
class SceneIR:
    source: SourceInfo
    nodes: dict = field(default_factory=dict)  # id -> IRNode (insertion ordered)
    joints: list = field(default_factory=list)
    world_physics: IRWorldPhysics | None = None
    has_physics: bool = False
    warnings: list = field(default_factory=list)

    def add(self, node: IRNode) -> IRNode:
        if node.id in self.nodes:
            raise ValueError(f"duplicate IR node id '{node.id}'")
        self.nodes[node.id] = node
        return node

    def roots(self):
        return [n for n in self.nodes.values() if n.parent is None or n.parent not in self.nodes]

    def world_transforms(self) -> dict:
        out: dict = {}

        def get(nid):
            if nid in out:
                return out[nid]
            n = self.nodes[nid]
            out[nid] = (
                n.local
                if (n.parent is None or n.parent not in self.nodes)
                else get(n.parent) @ n.local
            )
            return out[nid]

        for nid in self.nodes:
            get(nid)
        return out


def translation(p) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = p
    return t


def apply_lobs_annotations(node: IRNode, extras: dict | None) -> None:
    """Honor ``lobs:*`` annotations authored in the source tool (glTF extras, USD custom attributes,
    IFC property set ``Pset_LoBS``, SDF/URDF elements). These are *source* physics data."""
    if not extras:
        return
    ex = {str(k): v for k, v in extras.items() if str(k).startswith("lobs:")}
    if not ex:
        return
    node.physics = node.physics or IRPhysics()
    if "lobs:motion" in ex:
        node.physics.rigid = str(ex["lobs:motion"]).lower() == "dynamic"
    if "lobs:mass" in ex:
        node.physics.mass = float(ex["lobs:mass"])
    if "lobs:collision" in ex:
        node.physics.collision_hint = str(ex["lobs:collision"]).lower()
    if "lobs:material" in ex:
        node.material = IRMaterial(name=str(ex["lobs:material"]), raw={"annotated": True})
    if "lobs:tags" in ex:
        tags = ex["lobs:tags"]
        tags = [t.strip() for t in tags.split(",")] if isinstance(tags, str) else list(tags)
        node.tags.extend(t for t in tags if t)


def bake_mesh_node(node_id: str, name: str, vertices_world, faces, *, parent=None, **kw) -> IRNode:
    """A mesh element given in world coordinates: node frame = translation to the bbox centre."""
    v = np.asarray(vertices_world, dtype=float)
    centre = 0.5 * (v.min(axis=0) + v.max(axis=0))
    geom = IRGeom(kind="mesh", vertices=v - centre, faces=np.asarray(faces, dtype=int))
    return IRNode(
        id=node_id, name=name, parent=parent, local=translation(centre), geoms=[geom], **kw
    )
