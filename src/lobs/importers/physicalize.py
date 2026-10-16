"""Physicalize: turn a normalized SceneIR into a WorldSpec, logging every decision (plan 11.5).

Precedence for every field:  override > source > profile rule > mode / heuristic > profile default.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field

import numpy as np

from .. import math3d as m3
from ..assets import store_mesh
from ..errors import ImportFailure
from ..schema import (
    ActorSpec,
    ActuatorSpec,
    BodySpec,
    Box,
    Capsule,
    ConvexMesh,
    ConvexSet,
    Cylinder,
    GeomSpec,
    InertialSpec,
    JointSpec,
    MaterialSpec,
    Plane,
    Pose,
    SourceRef,
    Sphere,
    TriangleMesh,
    WorldSpec,
)
from . import geometry_fit as gf
from .decisions import DecisionLog, ImportReport, Overrides, load_materials
from .ir import IRGeom, IRNode, SceneIR

DEFAULT_DT = 0.002


def sanitize(name: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.\-]", "_", str(name or "")).strip("_.-")
    return s or fallback


class Names:
    def __init__(self, reserved=()):
        self.used = set(reserved)

    def unique(self, base: str) -> str:
        if base not in self.used:
            self.used.add(base)
            return base
        k = 2
        while f"{base}_{k}" in self.used:
            k += 1
        self.used.add(f"{base}_{k}")
        return f"{base}_{k}"


@dataclass
class Element:
    node: IRNode
    world: np.ndarray
    bbox: tuple  # (lo, hi) world
    rule: dict = field(default_factory=dict)
    ov: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    skipped: bool = False
    motion: str = "fixed"
    articulated: bool = False
    material: dict = field(default_factory=dict)
    mat_name: str = ""


# ------------------------------------------------------------------ helpers
def _glob(pattern, value) -> bool:
    return value is not None and fnmatch.fnmatchcase(str(value).lower(), str(pattern).lower())


def _match(m: dict, node: IRNode, tags: list, mat_name: str) -> bool:
    for key, want in m.items():
        wants = want if isinstance(want, (list, tuple)) else [want]
        if key == "class":
            ok = any(_glob(w, node.semantic_class) for w in wants)
        elif key == "name":
            ok = any(_glob(w, node.name) for w in wants)
        elif key == "tag":
            ok = any(_glob(w, t) for w in wants for t in tags)
        elif key == "material":
            ok = any(_glob(w, mat_name) for w in wants)
        elif key == "source_id":
            ok = any(_glob(w, node.id) for w in wants)
        else:
            raise ImportFailure(f"unknown rule match key '{key}'")
        if not ok:
            return False
    return True


def rule_values(rules: list, node: IRNode, tags: list) -> dict:
    out: dict = {}
    mat_name = node.material.name if node.material else ""
    for r in rules or []:
        if _match(r.get("match", {}), node, tags, mat_name or ""):
            for k, v in r.items():
                if k != "match" and k not in out:
                    out[k] = v
    return out


def _bbox(ir: SceneIR, nid: str, world: np.ndarray):
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in ir.nodes[nid].geoms:
        t = world @ g.transform
        if g.vertices is not None and len(g.vertices):
            pts = (t @ np.c_[g.vertices, np.ones(len(g.vertices))].T)[:3].T
        elif g.kind == "plane":
            pts = np.array([t[:3, 3]])
        else:
            half = {"box": np.array(g.size) / 2 if g.size else None,
                    "sphere": np.full(3, g.radius or 0.0),
                    "cylinder": np.array([g.radius or 0, g.radius or 0, (g.length or 0) / 2]),
                    "capsule": np.array([g.radius or 0, g.radius or 0, (g.length or 0) / 2 + (g.radius or 0)])}[g.kind]  # fmt: skip
            corners = (
                np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                * half
            )
            pts = (t[:3, :3] @ corners.T).T + t[:3, 3]
        lo, hi = np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))
    return lo, hi


# ------------------------------------------------------------------ materials
def resolve_material(el: Element, mats: dict, default: str, log: DecisionLog) -> None:
    node, ov, rule = el.node, el.ov, el.rule

    def lookup_by_keyword(text):
        t = (text or "").lower()
        if not t:
            return None
        for name, m in mats.items():
            if name in t.replace(" ", "_") or any(k in t for k in m.get("keywords", [])):
                if name != "generic":
                    return name
        return None

    name, prov, conf, why = (
        None,
        "default",
        "low",
        f"no material information; profile default '{default}'",
    )
    if ov.get("material"):
        name, prov, conf, why = ov["material"], "override", "high", "material given in overrides"
    elif node.material and node.material.raw.get("annotated") and node.material.name:
        name, prov, conf, why = (
            node.material.name,
            "source",
            "high",
            "lobs:material annotation in the source",
        )
    elif rule.get("material"):
        name, prov, conf, why = rule["material"], "rule", "high", "profile rule"
    else:
        for text in (node.material.name if node.material else None, node.name, node.semantic_class):
            hit = lookup_by_keyword(text)
            if hit:
                name, prov, conf, why = (
                    hit,
                    "material-library",
                    "medium",
                    f"keyword match on '{text}'",
                )
                break
    if name is None:
        name = default
    base = dict(mats.get(name) or mats.get("generic"))
    if name not in mats:
        why += f" (unknown material '{name}', using generic values)"
        conf = "low"
    values = {
        "density": base.get("density", 1000.0),
        "friction": base.get("friction", 0.6),
        "restitution": base.get("restitution", 0.1),
    }
    # explicit numbers in the source beat library numbers
    if node.material:
        for k in ("density", "friction", "restitution"):
            if getattr(node.material, k) is not None:
                values[k] = getattr(node.material, k)
                log.add(
                    node.id, f"material.{k}", values[k], "source", "high", "stated by the source"
                )
    for k in ("density", "friction", "restitution"):
        if ov.get(k) is not None:
            values[k] = float(ov[k])
            log.add(node.id, f"material.{k}", values[k], "override", "high", "overrides.yaml")
    log.add(node.id, "material", name, prov, conf, why, review=(prov == "default"))
    el.material, el.mat_name = values, name


# ------------------------------------------------------------------ motion
def decide_motion(
    el: Element, mode: str, prof: dict, mats_est_mass, log: DecisionLog, is_ground: bool
) -> None:
    node, ov, rule = el.node, el.ov, el.rule
    auto = prof["auto"]
    if ov.get("motion"):
        el.motion, why = ov["motion"], "overrides.yaml"
        log.add(node.id, "motion", el.motion, "override", "high", why)
        return
    if node.physics and node.physics.rigid is not None:
        el.motion = "dynamic" if node.physics.rigid else "fixed"
        log.add(node.id, "motion", el.motion, "source", "high", "stated by the source")
        return
    if rule.get("motion"):
        el.motion = rule["motion"]
        log.add(node.id, "motion", el.motion, "rule", "high", "profile rule")
        return
    if is_ground:
        el.motion = "fixed"
        log.add(node.id, "motion", "fixed", "heuristic", "high", "ground support is always fixed")
        return
    if mode == "dynamic":
        el.motion = "dynamic"
        log.add(
            node.id,
            "motion",
            "dynamic",
            "mode",
            "high",
            "--motion dynamic: everything not fixed by the source / rule / ground moves",
        )
        return
    cls = node.semantic_class or ""
    if any(_glob(c, cls) for c in auto["fixed_classes"]):
        el.motion = "fixed"
        log.add(node.id, "motion", "fixed", "heuristic", "high", f"class '{cls}' is structural")
        return
    if any(_glob(c, cls) for c in auto["dynamic_classes"]):
        el.motion = "dynamic"
        log.add(node.id, "motion", "dynamic", "heuristic", "high", f"class '{cls}' is movable")
        return
    text = " ".join(filter(None, [node.name, el.mat_name]))
    if any(k in text.lower() for k in auto["fixed_keywords"]):
        el.motion = "fixed"
        log.add(
            node.id,
            "motion",
            "fixed",
            "heuristic",
            "medium",
            f"name suggests structure ('{node.name}')",
        )
        return
    if any(k in text.lower() for k in auto["dynamic_keywords"]):
        el.motion = "dynamic"
        log.add(
            node.id,
            "motion",
            "dynamic",
            "heuristic",
            "medium",
            f"name suggests a movable object ('{node.name}')",
            review=True,
        )
        return
    lo, hi = el.bbox
    ext, mass = float(np.max(hi - lo)), mats_est_mass(el)
    if ext >= auto["fixed_min_extent"] or mass >= auto["fixed_min_mass"]:
        el.motion = "fixed"
        log.add(
            node.id,
            "motion",
            "fixed",
            "heuristic",
            "medium",
            f"large (extent {ext:.2f} m, ~{mass:.0f} kg)",
        )
    elif ext <= auto["dynamic_max_extent"] and mass <= auto["dynamic_max_mass"]:
        el.motion = "dynamic"
        log.add(
            node.id,
            "motion",
            "dynamic",
            "heuristic",
            "low",
            f"small and light (extent {ext:.2f} m, ~{mass:.0f} kg)",
            review=True,
        )
    else:
        el.motion = "fixed"
        log.add(
            node.id,
            "motion",
            "fixed",
            "heuristic",
            "low",
            "no rule matched; defaulting to fixed",
            review=True,
        )


# ------------------------------------------------------------------ geometry & mass
def _world_vertices(g: IRGeom) -> np.ndarray:
    return (g.transform @ np.c_[g.vertices, np.ones(len(g.vertices))].T)[:3].T


def _geom_volume_estimate(g: IRGeom) -> float:
    if g.kind == "mesh":
        return gf.mesh_props(_world_vertices(g), g.faces).volume
    if g.kind == "plane":
        return 0.0
    return gf.primitive_props(g.kind, g.size, g.radius, g.length)[0]


def _mass_parts(el: Element, density: float, source_mass: float | None):
    """[(mass, com, inertia about com)] in the node frame, scaled to the source mass if given."""
    parts, notes = [], []
    for g in el.node.geoms:
        if g.role == "visual":
            continue
        if g.kind == "mesh":
            mp = gf.mesh_props(_world_vertices(g), g.faces)
            parts.append((mp.volume, mp.centroid, mp.inertia_unit))
            if not mp.exact:
                notes.append("mesh is not watertight: mass properties from its convex hull")
        elif g.kind == "plane":
            continue
        else:
            vol, inertia = gf.primitive_props(g.kind, g.size, g.radius, g.length)
            r, c = g.transform[:3, :3], g.transform[:3, 3]
            parts.append((vol, c, r @ inertia @ r.T))
    total_vol = sum(p[0] for p in parts)
    if total_vol <= 0:
        return [], notes + ["no volume"], 0.0
    mass = source_mass if source_mass is not None else density * total_vol
    scale = mass / total_vol
    return [(p[0] * scale, p[1], p[2] * scale) for p in parts], notes, mass


def _unit_quat_pose(t: np.ndarray) -> Pose:
    pos, quat = gf.pose_from_matrix(t)
    return Pose(pos=tuple(float(x) for x in pos), quat=tuple(float(x) for x in quat))


def _build_geoms(
    el: Element,
    prof: dict,
    assets_dir,
    log: DecisionLog,
    fixed: bool,
    base_t: np.ndarray,
    ground_stats: dict,
):
    """GeomSpecs for one element, expressed relative to ``base_t`` (body frame in element-node frame)."""
    node = el.node
    params = prof["collision_params"]
    hint = (
        el.ov.get("collision")
        or (node.physics.collision_hint if node.physics else None)
        or el.rule.get("collision")
    )
    hint_prov = (
        "override"
        if el.ov.get("collision")
        else ("source" if node.physics and node.physics.collision_hint else "rule")
    )
    strategy = hint or prof.get("collision", "auto")
    geoms: list[GeomSpec] = []
    mat = MaterialSpec(
        name=el.mat_name,
        friction=el.material["friction"],
        restitution=el.material["restitution"],
        density=el.material["density"],
    )
    rgba = tuple(node.material.color) if node.material and node.material.color else None
    collide = not (node.physics and node.physics.collision_enabled is False) and strategy != "none"
    used_strategies = []
    for g in node.geoms:
        pose_t = np.linalg.inv(base_t) @ g.transform if base_t is not None else g.transform
        pose = _unit_quat_pose(pose_t)
        want_visual = prof.get("visual", "mesh") == "mesh" and g.kind == "mesh"
        if g.kind == "mesh":
            if want_visual and g.role != "collision":
                rel = store_mesh(assets_dir, g.vertices, g.faces)
                geoms.append(
                    GeomSpec(
                        shape=TriangleMesh(file=rel),
                        pose=pose,
                        role="visual",
                        rgba=rgba,
                        source_id=node.id,
                    )
                )
                ground_stats["triangles"] += len(g.faces)
            if collide and g.role != "visual":
                res = gf.fit_collision(g.vertices, g.faces, strategy, params, fixed)
                used_strategies.append(res.strategy)
                log.add(node.id, "collision", res.strategy, hint_prov if hint else "heuristic",
                        res.confidence, res.note, review=res.review)  # fmt: skip
                for sh in res.shapes:
                    if sh.kind == "box":
                        t = pose_t @ _rt(sh.rot, sh.center)
                        geoms.append(
                            GeomSpec(
                                shape=Box(size=tuple(float(x) for x in sh.size)),
                                pose=_unit_quat_pose(t),
                                material=mat,
                                role="collision",
                                source_id=node.id,
                            )
                        )
                    else:
                        rel = store_mesh(assets_dir, sh.vertices, sh.faces)
                        shape = (
                            TriangleMesh(file=rel) if sh.kind == "mesh" else ConvexMesh(file=rel)
                        )
                        geoms.append(
                            GeomSpec(
                                shape=shape,
                                pose=pose,
                                material=mat,
                                role="collision",
                                source_id=node.id,
                            )
                        )
            continue
        if not collide or g.role == "visual":
            continue
        shape = _primitive_shape(g)  # fmt: skip
        geoms.append(
            GeomSpec(
                shape=shape, pose=pose, material=mat, role="both", rgba=rgba, source_id=node.id
            )
        )
        log.add(
            node.id, "collision", "primitive", "source", "high", f"exact {g.kind} from the source"
        )
    # merge the collision hulls of one body so decompositions become a ConvexSet
    return _group_hulls(geoms)


def _primitive_shape(g: IRGeom):
    if g.kind == "box":
        return Box(size=tuple(float(x) for x in g.size))
    if g.kind == "sphere":
        return Sphere(radius=float(g.radius))
    if g.kind == "cylinder":
        return Cylinder(radius=float(g.radius), length=float(g.length))
    if g.kind == "capsule":
        return Capsule(radius=float(g.radius), length=float(g.length))
    if g.kind == "plane":
        return Plane()
    raise ImportFailure(f"unsupported primitive geometry '{g.kind}'")


def _rt(rot, centre) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, centre
    return t


def _group_hulls(geoms: list[GeomSpec]) -> list[GeomSpec]:
    """Consecutive ConvexMesh collision geoms sharing pose/source become one ConvexSet."""
    out, buf = [], []

    def flush():
        if not buf:
            return
        if len(buf) == 1:
            out.append(buf[0])
        else:
            first = buf[0]
            out.append(GeomSpec(shape=ConvexSet(files=[b.shape.file for b in buf]), pose=first.pose, material=first.material,
                                role="collision", source_id=first.source_id))  # fmt: skip
        buf.clear()

    for g in geoms:
        if (
            g.role == "collision"
            and g.shape.type == "convex_mesh"
            and (not buf or (buf[0].pose == g.pose and buf[0].source_id == g.source_id))
        ):
            buf.append(g)
        else:
            flush()
            out.append(g)
    flush()
    return out


def _explicit_inertial(el: Element, source_mass, log: DecisionLog, dynamic: bool) -> InertialSpec:
    node = el.node
    p = node.physics
    if p and p.mass is not None:
        source_mass = p.mass
        log.add(node.id, "mass", p.mass, "source", "high", "stated by the source")
    if el.ov.get("mass") is not None:
        source_mass = float(el.ov["mass"])
        log.add(node.id, "mass", source_mass, "override", "high", "overrides.yaml")
    if not dynamic:
        return InertialSpec(mode="explicit", mass=0.0, com=(0, 0, 0), inertia=((0, 0, 0),) * 3)
    parts, notes, mass = _mass_parts(el, el.material["density"], source_mass)
    prov = "source" if source_mass is not None else "material-library"
    if (
        not parts
    ):  # geometry-less or flat dynamic body: tiny placeholder mass keeps the solver happy
        mass = source_mass if source_mass is not None else 1e-3
        log.add(
            node.id,
            "mass",
            mass,
            "default",
            "low",
            "no volume to integrate: placeholder mass/inertia",
            review=True,
        )
        i = np.eye(3) * max(mass * 1e-6, 1e-9)
        return InertialSpec(
            mode="explicit", mass=float(mass), com=(0, 0, 0), inertia=tuple(map(tuple, i.tolist()))
        )
    m, com, inertia = gf.combine_parts(parts)
    if p and p.com is not None:
        com = np.asarray(p.com, float)
    if p and p.inertia is not None:
        inertia = np.asarray(p.inertia, float)
    if source_mass is None:
        log.add(
            node.id,
            "mass",
            float(m),
            prov,
            "medium",
            f"density {el.material['density']:g} kg/m^3 x volume ({el.mat_name})",
        )
    for n in notes:
        log.add(node.id, "mass", float(m), "heuristic", "low", n, review=True)
    return InertialSpec(
        mode="explicit",
        mass=float(m),
        com=tuple(float(x) for x in com),
        inertia=tuple(tuple(float(x) for x in row) for row in inertia),
    )


# ------------------------------------------------------------------ main
@dataclass
class Result:
    spec: WorldSpec
    report: ImportReport
    names: dict  # source id -> actor name


def physicalize(ir: SceneIR, *, profile: dict, motion: str | None, overrides: Overrides, assets_dir, log: DecisionLog,
                name: str = "imported", georef: dict | None = None, actuate=None) -> Result:  # fmt: skip
    if motion not in ("auto", "dynamic"):
        raise ImportFailure(
            "motion mode is required: pass motion='auto' (heuristic guess) or motion='dynamic' (everything movable)"
        )
    mats = load_materials(profile)
    _attach_world_joints(ir)
    world_t = ir.world_transforms()
    default_mat = profile.get("default_material", "generic")

    # ---- elements
    elements: dict[str, Element] = {}
    joint_nodes = {j.parent_node for j in ir.joints} | {j.child_node for j in ir.joints}
    for nid, node in ir.nodes.items():
        if not node.geoms and nid not in joint_nodes:
            continue
        w = world_t[nid]
        tags = list(node.tags) + ([f"class:{node.semantic_class}"] if node.semantic_class else [])
        rule = rule_values(profile.get("rules"), node, tags)
        ov = overrides.get(nid)
        el = Element(
            node, w, _bbox(ir, nid, w) if node.geoms else (w[:3, 3], w[:3, 3]), rule, ov, tags
        )
        if rule.get("skip") or ov.get("skip"):
            el.skipped = True
            log.add(
                nid,
                "skip",
                True,
                "override" if ov.get("skip") else "rule",
                "high",
                "excluded from the world",
            )
        for t in list(rule.get("tags", [])) + list(ov.get("tags", [])):
            el.tags.append(t)
        elements[nid] = el
        if not el.skipped:
            resolve_material(el, mats, default_mat, log)

    live = {k: e for k, e in elements.items() if not e.skipped}
    if not live:
        raise ImportFailure("nothing to import: the source has no (non-skipped) geometry")

    # ---- articulated components
    parent_of = {
        j.child_node: j.parent_node
        for j in ir.joints
        if j.child_node in live and j.parent_node in live
    }
    comp_nodes: dict[str, list[str]] = {}
    root_of = {}
    for nid in live:
        r = nid
        seen = {r}
        while r in parent_of:
            r = parent_of[r]
            if r in seen:
                raise ImportFailure(
                    f"joint loop through '{r}': closed kinematic chains are not supported"
                )
            seen.add(r)
        root_of[nid] = r
    for nid in live:
        comp_nodes.setdefault(root_of[nid], []).append(nid)
    for members in comp_nodes.values():
        if len(members) > 1:
            for m in members:
                live[m].articulated = True

    # ---- ground detection (non-articulated elements)
    geo = [e for e in live.values() if e.node.geoms] or list(live.values())
    all_lo = np.min([e.bbox[0] for e in geo], axis=0)
    all_hi = np.max([e.bbox[1] for e in geo], axis=0)
    scene_area = max(float((all_hi[0] - all_lo[0]) * (all_hi[1] - all_lo[1])), 1e-9)
    gp = profile["ground"]
    ground_ids = set()
    for nid, e in live.items():
        if e.articulated:
            continue
        lo, hi = e.bbox
        footprint = float((hi[0] - lo[0]) * (hi[1] - lo[1]))
        by_class = any(_glob(c, e.node.semantic_class) for c in gp["classes"])
        by_name = any(k in e.node.name.lower() for k in gp["keywords"])
        by_geom = (
            abs(lo[2] - all_lo[2]) <= gp["tol_z"]
            and footprint / scene_area >= gp["min_footprint_fraction"]
            and len(live) > 1
        )
        if by_class or by_name or by_geom:
            ground_ids.add(nid)

    def est_mass(e: Element) -> float:
        return float(e.material.get("density", 1000.0)) * sum(
            _geom_volume_estimate(g) for g in e.node.geoms if g.role != "visual"
        )

    for root, members in comp_nodes.items():
        r = live[root]
        decide_motion(r, motion, profile, est_mass, log, root in ground_ids)
        for m in members:
            live[m].motion = r.motion

    joints_by_child = {j.child_node: j for j in ir.joints}

    # ---- build actors
    names = Names(reserved={"world"})
    spec = WorldSpec(name=name)
    spec.contact.default_material = MaterialSpec(name=default_mat, friction=float(mats.get(default_mat, mats["generic"])["friction"]),
                                                 restitution=float(mats.get(default_mat, mats["generic"])["restitution"]),
                                                 density=float(mats.get(default_mat, mats["generic"])["density"]))  # fmt: skip
    stats = {"triangles": 0, "geoms": 0}
    actors_by_id: dict[str, str] = {}
    merged_fixed = profile.get("merge_fixed", "none") == "all"
    static_geoms: list[GeomSpec] = []
    ug = _UnsupportedJoints(profile.get("on_unsupported_joint", "error"), log)

    for root, members in comp_nodes.items():
        r = live[root]
        node = r.node
        if node.id.endswith("@world") and len(members) > 1:
            node = live[
                next(m for m in members if m != root)
            ].node  # name/attribute the actor after the real body
        if len(members) == 1 and r.motion == "fixed" and merged_fixed and not r.articulated:
            geoms = _build_geoms(r, profile, assets_dir, log, True, None, stats)
            for g in geoms:
                g.pose = _unit_quat_pose(r.world @ _matrix_from_pose(g.pose))
            static_geoms.extend(geoms)
            actors_by_id[node.id] = "static_scene"
            continue
        actor_name = names.unique(
            sanitize(
                node.name,
                sanitize(node.semantic_class or "", "element") + "_" + sanitize(node.id[-6:], "x"),
            )
        )
        actors_by_id[node.id] = actor_name
        actor = ActorSpec(
            name=actor_name, root="free" if r.motion == "dynamic" else "fixed",
            pose=_unit_quat_pose(r.world),
            tags=sorted(set(r.tags)),
            source=SourceRef(format=ir.source.format, source_id=node.id, source_path=ir.source.path, semantic_class=node.semantic_class),
        )  # fmt: skip
        order = _bfs_order(root, members, parent_of)
        body_names: dict[str, str] = {}
        bnames = Names()
        for nid in order:
            e = live[nid]
            bname = "body" if len(members) == 1 else bnames.unique(sanitize(e.node.name, "link"))
            body_names[nid] = bname
            geoms = _build_geoms(
                e, profile, assets_dir, log, r.motion == "fixed" and nid == root, np.eye(4), stats
            )
            inertial = _explicit_inertial(
                e, None, log, dynamic=(r.motion == "dynamic" or nid != root)
            )
            if nid == root:
                pose = Pose()
                parent = None
            else:
                pw = live[parent_of[nid]].world
                pose = _unit_quat_pose(np.linalg.inv(pw) @ e.world)
                parent = body_names[parent_of[nid]]
            actor.bodies.append(BodySpec(name=bname, parent=parent, pose=pose, geoms=geoms, inertial=inertial,
                                         tags=sorted(set(e.tags)), source=SourceRef(format=ir.source.format, source_id=e.node.id,
                                                                                    source_path=ir.source.path, semantic_class=e.node.semantic_class)))  # fmt: skip
            stats["geoms"] += len(geoms)
        jn = Names(reserved=set(body_names.values()))
        for nid in order:
            j = joints_by_child.get(nid)
            if j is not None and j.parent_node in body_names:
                js = ug.convert(j, jn, body_names, actor)
                if js is not None:
                    actor.joints.append(js)
                    _maybe_actuator(actor, js, j, actuate, log)
        # bodies connected only by a dropped/fixed policy must still have joints
        have = {j.child for j in actor.joints}
        for b in actor.bodies:
            if b.parent is not None and b.name not in have:
                actor.joints.append(
                    JointSpec(
                        name=jn.unique(f"{b.name}_weld"),
                        type="fixed",
                        parent=b.parent,
                        child=b.name,
                    )
                )
        spec.actors.append(actor)

    if static_geoms:
        spec.actors.append(ActorSpec(name=names.unique("static_scene"), root="fixed", bodies=[
            BodySpec(name="body", geoms=static_geoms, inertial=InertialSpec(mode="explicit", mass=0.0, inertia=((0, 0, 0),) * 3))]))  # fmt: skip
        stats["geoms"] += len(static_geoms)
        log.add(
            "scene",
            "merge_fixed",
            len(static_geoms),
            "override",
            "high",
            "all fixed elements merged into one static body",
        )

    # ---- ground
    if profile.get("add_ground", "auto") == "auto" and not ground_ids:
        spec.actors.append(ActorSpec(name=names.unique("ground"), root="fixed", pose=Pose(pos=(0.0, 0.0, float(all_lo[2]))), bodies=[
            BodySpec(name="body", geoms=[GeomSpec(shape=Plane(), material=MaterialSpec(name=default_mat, friction=spec.contact.default_material.friction,
                                                                                   restitution=spec.contact.default_material.restitution))],
                     inertial=InertialSpec(mode="explicit", mass=0.0, inertia=((0, 0, 0),) * 3))]))  # fmt: skip
        log.add(
            "scene",
            "ground",
            "plane",
            "heuristic",
            "medium",
            f"no ground support found: added a ground plane at z={all_lo[2]:.3f}",
            review=True,
        )
    else:
        for gid in sorted(ground_ids):
            log.add(gid, "ground", True, "heuristic", "high", "treated as ground support (fixed)")

    # ---- world physics
    wp = ir.world_physics
    if wp and wp.gravity is not None:
        spec.gravity = tuple(float(x) for x in wp.gravity)
        log.add("scene", "gravity", list(spec.gravity), "source", "high", "stated by the source")
    if wp and wp.timestep and 1e-4 <= wp.timestep <= 0.02:
        spec.timing.dt = float(wp.timestep)
        log.add("scene", "dt", spec.timing.dt, "source", "high", "stated by the source")
    else:
        spec.timing.dt = DEFAULT_DT
    spec.georeference = None
    if georef:
        from ..schema.world import Georeference

        spec.georeference = Georeference(crs=georef.get("crs"), offset_m=tuple(georef["offset_m"]))

    report = ImportReport(source=ir.source.path, format=ir.source.format, profile=profile.get("profile", "custom"), motion=motion,
                          decisions=log, warnings=list(ir.warnings), georeference=georef)  # fmt: skip
    dyn = sum(1 for a in spec.actors if a.root == "free")
    report.counts = {
        "actors": len(spec.actors),
        "dynamic": dyn,
        "fixed": len(spec.actors) - dyn,
        "geoms": stats["geoms"],
        "triangles": stats["triangles"],
    }
    for u in overrides.unused():
        report.warnings.append(f"override for unknown source id '{u}' was not used")
    return Result(spec, report, actors_by_id)


def _attach_world_joints(ir: SceneIR) -> None:
    """A joint whose parent is the world gets a synthetic fixed, geometry-less base body at the origin."""
    from .ir import IRPhysics

    for j in ir.joints:
        if j.parent_node is None:
            base = IRNode(
                id=f"{j.child_node}@world", name="world_base", physics=IRPhysics(rigid=False)
            )
            ir.nodes[base.id] = base
            j.parent_node = base.id


def _matrix_from_pose(p: Pose) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = m3.quat_to_mat(np.array(p.quat))
    t[:3, 3] = p.pos
    return t


def _bfs_order(root: str, members: list[str], parent_of: dict) -> list[str]:
    children: dict[str, list[str]] = {}
    for m in members:
        if m in parent_of:
            children.setdefault(parent_of[m], []).append(m)
    out, queue = [], [root]
    while queue:
        n = queue.pop(0)
        out.append(n)
        queue.extend(children.get(n, []))
    return out


class _UnsupportedJoints:
    def __init__(self, policy: str, log: DecisionLog):
        self.policy, self.log = policy, log

    def convert(self, j, jn: Names, body_names: dict, actor: ActorSpec):
        parent, child = body_names[j.parent_node], body_names[j.child_node]
        name = jn.unique(sanitize(j.name, "joint"))
        t = j.type
        if t in ("revolute", "continuous"):
            rng = tuple(j.limits) if (t == "revolute" and j.limits is not None) else None
            return JointSpec(
                name=name,
                type="hinge",
                parent=parent,
                child=child,
                pos=tuple(j.pos),
                axis=tuple(j.axis),
                range=rng,
                damping=j.damping,
            )
        if t == "prismatic":
            rng = tuple(j.limits) if j.limits is not None else None
            return JointSpec(
                name=name,
                type="slide",
                parent=parent,
                child=child,
                pos=tuple(j.pos),
                axis=tuple(j.axis),
                range=rng,
                damping=j.damping,
            )
        if t == "fixed":
            return JointSpec(name=name, type="fixed", parent=parent, child=child)
        if t == "spherical":
            return JointSpec(
                name=name,
                type="ball",
                parent=parent,
                child=child,
                pos=tuple(j.pos),
                damping=j.damping,
            )
        if self.policy == "fix":
            self.log.add(
                j.id,
                "joint",
                "fixed",
                "heuristic",
                "low",
                f"unsupported joint type '{t}' replaced by a weld",
                review=True,
            )
            return JointSpec(name=name, type="fixed", parent=parent, child=child)
        if self.policy == "drop":
            self.log.add(
                j.id,
                "joint",
                "dropped",
                "heuristic",
                "low",
                f"unsupported joint type '{t}' dropped (bodies welded)",
                review=True,
            )
            return None
        raise ImportFailure(
            f"unsupported joint type '{t}' in joint '{j.name}' (set on_unsupported_joint to 'fix' or 'drop' to continue)"
        )


def _maybe_actuator(actor: ActorSpec, js: JointSpec, j, actuate, log: DecisionLog):
    if js.type not in ("hinge", "slide"):
        return
    if j.drive is not None:
        kp, kd = j.drive
        actor.actuators.append(
            ActuatorSpec(
                name=f"{js.name}_act",
                joint=js.name,
                kind="position",
                kp=float(kp),
                kd=float(kd),
                effort_limit=j.effort_limit,
            )
        )
        log.add(j.id, "actuator", "position", "source", "high", f"joint drive kp={kp:g} kd={kd:g}")
    elif actuate == "all" or (actuate and js.name in actuate) or (actuate and j.name in actuate):
        actor.actuators.append(
            ActuatorSpec(
                name=f"{js.name}_act", joint=js.name, kind="effort", effort_limit=j.effort_limit
            )
        )
        log.add(j.id, "actuator", "effort", "mode", "high", "actuate option")
