"""SDFormat reader (stdlib XML), documented subset of SDF >= 1.6.

Supported: worlds and models (nested models, ``<include>`` via ``model_paths=``), links with inertial /
visual / collision, geometry (box, sphere, cylinder, capsule, plane, mesh), surface friction (``mu``) and
restitution, joints (revolute, prismatic, fixed, ball) incl. joints to the world, ``<static>``, gravity and
step size. Poses are ``x y z roll pitch yaw`` relative to the enclosing model (or the frame named by
``relative_to`` if it is the model, the world or a link of the same model); joint poses are relative to the
child link unless ``relative_to`` says otherwise. Anything else (heightmaps, universal joints, ...) is
reported and skipped.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ... import math3d as m3
from ..ir import (
    IRGeom,
    IRJoint,
    IRNode,
    IRPhysics,
    IRWorldPhysics,
    SceneIR,
    SourceInfo,
    apply_lobs_annotations,
)
from .urdf import floats, load_mesh_file, rpy_matrix

_JOINT_TYPES = {
    "revolute": "revolute",
    "prismatic": "prismatic",
    "fixed": "fixed",
    "ball": "spherical",
    "continuous": "continuous",
}


def pose_matrix(el) -> np.ndarray:
    t = np.eye(4)
    if el is None or not (el.text or "").strip():
        return t
    v = floats(el.text.strip())
    v = v + [0.0] * (6 - len(v))
    r = v[3:6]
    if str(el.get("degrees", "false")).lower() == "true":
        r = [np.radians(x) for x in r]
    t[:3, 3] = v[:3]
    t[:3, :3] = rpy_matrix(*r)
    return t


def _text(el, path, default=None):
    f = el.find(path)
    return f.text.strip() if f is not None and f.text else default


def read(path, *, model_paths=None, **_):
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != "sdf":
        raise ValueError("not an SDF file (root element is not <sdf>)")
    src = SourceInfo(path=str(path), format="sdf", meters_per_unit=1.0, up_axis="z")
    ir = SceneIR(src)
    ir.has_physics = True
    dirs = [Path(d) for d in (model_paths or [])] + [path.parent, path.parent.parent]
    world_el = root.find("world")
    scope = world_el if world_el is not None else root
    if world_el is not None:
        g = _text(world_el, "gravity")
        step = _text(world_el, "physics/max_step_size")
        ir.world_physics = IRWorldPhysics(
            gravity=tuple(floats(g, 3)) if g else None, timestep=float(step) if step else None
        )
    ctx = _Ctx(ir, dirs, path.parent)
    for child in scope:
        if child.tag == "model":
            ctx.model(child, np.eye(4), prefix="", static=False)
        elif child.tag == "include":
            ctx.include(child, np.eye(4), prefix="")
        elif child.tag in (
            "light",
            "gui",
            "scene",
            "physics",
            "gravity",
            "plugin",
            "atmosphere",
            "magnetic_field",
            "spherical_coordinates",
        ):
            continue
        elif child.tag not in ("model", "include"):
            pass
    if not ir.nodes:
        ir.warnings.append("no links found")
    return ir


class _Ctx:
    def __init__(self, ir: SceneIR, dirs, base: Path):
        self.ir, self.dirs, self.base = ir, dirs, base

    def include(self, el, parent_world, prefix):
        uri = _text(el, "uri", "")
        name = uri.replace("model://", "").strip("/")
        for d in self.dirs:
            for cand in (d / name / "model.sdf", d / f"{name}.sdf", d / name):
                if cand.is_file():
                    model_root = ET.parse(cand).getroot()
                    model_el = model_root.find("model")
                    if model_el is not None:
                        nm = _text(el, "name")
                        if nm:
                            model_el.set("name", nm)
                        pose = pose_matrix(el.find("pose"))
                        static = _text(el, "static", "false").lower() == "true"
                        sub = _Ctx(self.ir, [cand.parent, *self.dirs], cand.parent)
                        sub.model(model_el, parent_world @ pose, prefix, static)
                        return
        self.ir.warnings.append(f"include '{uri}' not found (pass model_paths=)")

    def model(self, el, parent_world, prefix, static):
        name = el.get("name", "model")
        full = f"{prefix}{name}"
        world = parent_world @ pose_matrix(el.find("pose"))
        static = static or _text(el, "static", "false").lower() == "true"
        links = {}
        n_links = len(el.findall("link"))
        for link in el.findall("link"):
            lname = link.get("name")
            display = (
                name if n_links == 1 else f"{name}_{lname}"
            )  # single-link models are named after the model
            links[lname] = self.link(link, world, full, static, display)
        for sub in el.findall("model"):
            self.model(sub, world, f"{full}/", static)
        for inc in el.findall("include"):
            self.include(inc, world, f"{full}/")
        for j in el.findall("joint"):
            self.joint(j, links, full)

    def link(self, el, model_world, full, static, display=None) -> IRNode:
        name = el.get("name")
        world = model_world @ pose_matrix(el.find("pose"))
        node = IRNode(
            id=f"{full}/{name}",
            name=display or name,
            local=world,
            tags=[f"group:{p}" for p in full.split("/")],
        )
        phys = IRPhysics(rigid=False if static else None)
        ine = el.find("inertial")
        if ine is not None and ine.find("mass") is not None:
            ip = pose_matrix(ine.find("pose"))
            phys.mass = float(_text(ine, "mass"))
            phys.com = tuple(ip[:3, 3])
            t = ine.find("inertia")
            if t is not None:
                g = lambda k: float(_text(t, k, "0"))  # noqa: E731
                tensor = np.array(
                    [
                        [g("ixx"), g("ixy"), g("ixz")],
                        [g("ixy"), g("iyy"), g("iyz")],
                        [g("ixz"), g("iyz"), g("izz")],
                    ]
                )
                phys.inertia = ip[:3, :3] @ tensor @ ip[:3, :3].T
        node.physics = phys if (phys.mass is not None or phys.rigid is not None) else None
        mu = None
        rest = None
        for tag, role in (("visual", "visual"), ("collision", "collision")):
            for ge in el.findall(tag):
                spec = self.geometry(ge.find("geometry"), f"link '{full}/{name}' {tag}")
                if spec is None:
                    continue
                if role == "collision":
                    mu_v = _text(ge, "surface/friction/ode/mu")
                    mu = float(mu_v) if mu_v and mu is None else mu
                    r_v = _text(ge, "surface/bounce/restitution_coefficient")
                    rest = float(r_v) if r_v and rest is None else rest
                node.geoms.append(IRGeom(kind=spec["kind"], transform=pose_matrix(ge.find("pose")), role=role,
                                         radius=spec.get("radius"), length=spec.get("length"), size=spec.get("size"),
                                         vertices=spec.get("vertices"), faces=spec.get("faces")))  # fmt: skip
        if mu is not None or rest is not None:
            from ..ir import IRMaterial

            node.material = IRMaterial(friction=mu, restitution=rest, raw={"annotated": False})
        lobs_el = el.find("lobs")
        if lobs_el is not None:
            apply_lobs_annotations(node, {f"lobs:{k}": v for k, v in lobs_el.attrib.items()})
        self.ir.add(node)
        return node

    def geometry(self, geom_el, where):
        if geom_el is None:
            return None
        for child in geom_el:
            t = child.tag
            if t == "box":
                return {"kind": "box", "size": tuple(floats(_text(child, "size"), 3))}
            if t == "sphere":
                return {"kind": "sphere", "radius": float(_text(child, "radius"))}
            if t in ("cylinder", "capsule"):
                return {
                    "kind": t,
                    "radius": float(_text(child, "radius")),
                    "length": float(_text(child, "length")),
                }
            if t == "plane":
                return {"kind": "plane"}
            if t == "mesh":
                uri = _text(child, "uri", "")
                path = None
                if uri.startswith("model://"):
                    rel = uri[len("model://") :]
                    path = next((d / rel for d in self.dirs if (d / rel).exists()), None)
                else:
                    path = Path(uri) if Path(uri).is_absolute() else self.base / uri
                if path is None or not path.exists():
                    self.ir.warnings.append(f"{where}: mesh '{uri}' not found; geometry skipped")
                    return None
                try:
                    data = load_mesh_file(path)
                except Exception as e:  # noqa: BLE001
                    self.ir.warnings.append(
                        f"{where}: cannot read mesh '{path.name}' ({type(e).__name__}); skipped"
                    )
                    return None
                if data is None:
                    return None
                scale = np.array(floats(_text(child, "scale"), 3, [1, 1, 1]))
                v, f = data
                return {
                    "kind": "mesh",
                    "vertices": v * scale,
                    "faces": f if np.prod(np.sign(scale)) > 0 else f[:, ::-1],
                }
            self.ir.warnings.append(f"{where}: geometry <{t}> is not supported; skipped")
            return None
        return None

    def joint(self, el, links, full):
        name, jtype = el.get("name"), el.get("type")
        if jtype not in _JOINT_TYPES:
            self.ir.warnings.append(
                f"joint '{full}/{name}': type '{jtype}' is not supported; skipped"
            )
            return
        parent_name, child_name = _text(el, "parent"), _text(el, "child")
        child = links.get(child_name)
        if child is None:
            self.ir.warnings.append(
                f"joint '{full}/{name}': unknown child link '{child_name}'; skipped"
            )
            return
        parent = None if parent_name == "world" else links.get(parent_name)
        if parent_name != "world" and parent is None:
            self.ir.warnings.append(
                f"joint '{full}/{name}': unknown parent link '{parent_name}'; skipped"
            )
            return
        rel = el.find("pose").get("relative_to") if el.find("pose") is not None else None
        jp = pose_matrix(el.find("pose"))
        if rel and rel not in (child_name, ""):
            if (
                rel in links
            ):  # joint pose given relative to another link: convert into the child link frame
                jp = np.linalg.inv(child.local) @ links[rel].local @ jp
            elif rel in ("world", "__model__"):
                jp = np.linalg.inv(child.local) @ jp
        axis_el = el.find("axis")
        axis = np.array([0.0, 0.0, 1.0])
        limits = None
        damping, effort = 0.0, None
        if axis_el is not None:
            axis = np.array(floats(_text(axis_el, "xyz"), 3, [0, 0, 1]))
            lo, hi = _text(axis_el, "limit/lower"), _text(axis_el, "limit/upper")
            if lo is not None and hi is not None and jtype in ("revolute", "prismatic"):
                a, b = float(lo), float(hi)
                if np.isfinite(a) and np.isfinite(b) and abs(a) < 1e8 and abs(b) < 1e8:
                    limits = (a, b)
            damping = float(_text(axis_el, "dynamics/damping", "0") or 0)
            ef = _text(axis_el, "limit/effort")
            effort = float(ef) if ef and float(ef) > 0 and np.isfinite(float(ef)) else None
            if _text(axis_el, "use_parent_model_frame", "false").lower() == "true":
                # axis is given in the model frame: bring it into the child link frame
                axis = child.local[:3, :3].T @ axis
        axis_child = jp[:3, :3] @ (axis / (np.linalg.norm(axis) or 1.0))
        self.ir.joints.append(IRJoint(
            id=f"{full}/{name}", name=name, type=_JOINT_TYPES[jtype],
            parent_node=None if parent is None else parent.id, child_node=child.id,
            pos=tuple(jp[:3, 3]), axis=tuple(axis_child), limits=limits, damping=damping, effort_limit=effort,
        ))  # fmt: skip
        _ = m3
