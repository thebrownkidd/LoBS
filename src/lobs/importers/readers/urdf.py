"""URDF reader (stdlib XML). Robots: links with inertials / visuals / collisions and a joint tree.

Conventions: metres, Z-up, rpy = fixed-axis roll-pitch-yaw (R = Rz(yaw) Ry(pitch) Rx(roll)); a link
frame is its parent joint frame, so joint axes and origins are already in the child link frame.
Meshes are resolved from ``package://`` and relative paths (``package_dirs=``) and loaded with trimesh;
unreadable formats (e.g. COLLADA without pycollada) are skipped with a warning.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ..ir import (
    IRGeom,
    IRJoint,
    IRMaterial,
    IRNode,
    IRPhysics,
    SceneIR,
    SourceInfo,
    apply_lobs_annotations,
)


def floats(text, n=None, default=None):
    if text is None:
        return default
    v = [float(x) for x in text.split()]
    if n is not None and len(v) != n:
        raise ValueError(f"expected {n} numbers, got '{text}'")
    return v


def rpy_matrix(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def origin_matrix(el) -> np.ndarray:
    t = np.eye(4)
    if el is not None:
        t[:3, 3] = floats(el.get("xyz"), 3, [0, 0, 0])
        t[:3, :3] = rpy_matrix(*floats(el.get("rpy"), 3, [0, 0, 0]))
    return t


def resolve_uri(uri: str, base: Path, package_dirs) -> Path | None:
    """package://pkg/rel, file://abs, or a path relative to the description file."""
    if uri.startswith("file://"):
        return Path(uri[7:])
    if uri.startswith("package://"):
        pkg, _, rel = uri[len("package://") :].partition("/")
        roots = []
        if isinstance(package_dirs, dict):
            roots += [Path(package_dirs[pkg])] if pkg in package_dirs else []
        elif package_dirs:
            roots += [Path(d) for d in package_dirs]
        cur = base
        for _ in range(5):  # ROS layout: <pkg>/urdf/robot.urdf -> package root is an ancestor
            roots += [cur, cur.parent / pkg]
            cur = cur.parent
        for r in roots:
            for cand in (r / pkg / rel, r / rel):
                if cand.exists():
                    return cand
        return None
    p = Path(uri)
    return p if p.is_absolute() else base / p


def load_mesh_file(path: Path):
    import trimesh

    m = trimesh.load(str(path), force="mesh")
    if not hasattr(m, "faces") or len(m.faces) == 0:
        return None
    return np.asarray(m.vertices, float), np.asarray(m.faces, int)


def parse_geometry(geom_el, base: Path, package_dirs, warnings: list, where: str):
    """-> dict(kind=..., ...) or None."""
    for child in geom_el:
        tag = child.tag
        if tag == "box":
            return {"kind": "box", "size": tuple(floats(child.get("size"), 3))}
        if tag == "sphere":
            return {"kind": "sphere", "radius": float(child.get("radius"))}
        if tag in ("cylinder", "capsule"):
            return {
                "kind": tag,
                "radius": float(child.get("radius")),
                "length": float(child.get("length")),
            }
        if tag == "plane":
            return {"kind": "plane"}
        if tag == "mesh":
            uri = child.get("filename") or child.get("uri")
            path = resolve_uri(uri, base, package_dirs)
            if path is None or not path.exists():
                warnings.append(
                    f"{where}: mesh '{uri}' not found (pass package_dirs=); geometry skipped"
                )
                return None
            try:
                data = load_mesh_file(path)
            except Exception as e:  # noqa: BLE001 - unreadable formats are reported, not fatal
                warnings.append(
                    f"{where}: cannot read mesh '{path.name}' ({type(e).__name__}); geometry skipped"
                )
                return None
            if data is None:
                warnings.append(f"{where}: mesh '{path.name}' has no triangles; geometry skipped")
                return None
            scale = np.array(floats(child.get("scale"), 3, [1, 1, 1]))
            v, f = data
            v = v * scale
            if np.prod(np.sign(scale)) < 0:
                f = f[:, ::-1]
            return {"kind": "mesh", "vertices": v, "faces": f}
    return None


def make_geom(spec: dict, transform: np.ndarray, role: str, material) -> IRGeom:
    return IRGeom(kind=spec["kind"], transform=transform, role=role, material=material,
                  radius=spec.get("radius"), length=spec.get("length"), size=spec.get("size"),
                  vertices=spec.get("vertices"), faces=spec.get("faces"))  # fmt: skip


def read(path, *, package_dirs=None, fixed_base=True, **_):
    path = Path(path)
    root = ET.parse(path).getroot()
    if root.tag != "robot":
        raise ValueError("not a URDF file (root element is not <robot>)")
    base = path.parent
    src = SourceInfo(path=str(path), format="urdf", meters_per_unit=1.0, up_axis="z")
    ir = SceneIR(src)
    ir.has_physics = True
    if root.find(".//{*}xacro") is not None or "xacro" in (root.get("xmlns:xacro") or ""):
        ir.warnings.append("xacro macros are not expanded; run `xacro` first")
    materials = {}
    for m in root.findall("material"):
        c = m.find("color")
        if c is not None and c.get("rgba"):
            materials[m.get("name")] = tuple(floats(c.get("rgba"), 4))

    links = {el.get("name"): el for el in root.findall("link")}
    joints = root.findall("joint")
    child_of = {j.find("child").get("link"): j for j in joints}
    for name, el in links.items():
        node = IRNode(id=name, name=name)
        parent_joint = child_of.get(name)
        if parent_joint is not None:
            node.parent = parent_joint.find("parent").get("link")
            node.local = origin_matrix(parent_joint.find("origin"))
        # inertial
        inertial = el.find("inertial")
        phys = IRPhysics()
        if inertial is not None and inertial.find("mass") is not None:
            io = origin_matrix(inertial.find("origin"))
            phys.mass = float(inertial.find("mass").get("value"))
            phys.com = tuple(io[:3, 3])
            ine = inertial.find("inertia")
            if ine is not None:
                v = {k: float(ine.get(k, 0.0)) for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
                tensor = np.array(
                    [
                        [v["ixx"], v["ixy"], v["ixz"]],
                        [v["ixy"], v["iyy"], v["iyz"]],
                        [v["ixz"], v["iyz"], v["izz"]],
                    ]
                )
                phys.inertia = io[:3, :3] @ tensor @ io[:3, :3].T
        if parent_joint is None:
            phys.rigid = (
                not fixed_base
            )  # the URDF root: fixed to the world unless a floating base is requested
        # geometry
        color = None
        for tag, role in (("visual", "visual"), ("collision", "collision")):
            for ge in el.findall(tag):
                geom_el = ge.find("geometry")
                if geom_el is None:
                    continue
                spec = parse_geometry(
                    geom_el, base, package_dirs, ir.warnings, f"link '{name}' {tag}"
                )
                if spec is None:
                    continue
                mat = None
                if role == "visual":
                    me = ge.find("material")
                    if me is not None:
                        c = me.find("color")
                        rgba = (
                            tuple(floats(c.get("rgba"), 4))
                            if c is not None and c.get("rgba")
                            else materials.get(me.get("name"))
                        )
                        if rgba:
                            color = color or rgba
                node.geoms.append(make_geom(spec, origin_matrix(ge.find("origin")), role, mat))
        node.physics = phys if (phys.mass is not None or phys.rigid is not None) else None
        if color is not None:
            node.material = IRMaterial(color=color, raw={"annotated": False})
        lobs_el = el.find("lobs")
        if lobs_el is not None:
            apply_lobs_annotations(node, {f"lobs:{k}": v for k, v in lobs_el.attrib.items()})
        ir.add(node)

    for j in joints:
        jtype = j.get("type")
        limit = j.find("limit")
        limits = None
        if (
            jtype == "revolute"
            and limit is not None
            and limit.get("lower") is not None
            and limit.get("upper") is not None
        ):
            limits = (float(limit.get("lower")), float(limit.get("upper")))
        elif jtype == "prismatic" and limit is not None and limit.get("lower") is not None:
            limits = (float(limit.get("lower")), float(limit.get("upper")))
        dyn = j.find("dynamics")
        ax = j.find("axis")
        axis = tuple(floats(ax.get("xyz"), 3)) if ax is not None else (1.0, 0.0, 0.0)
        n = np.linalg.norm(axis)
        axis = tuple(a / n for a in axis) if n > 0 else (1.0, 0.0, 0.0)
        ir.joints.append(IRJoint(
            id=j.get("name"), name=j.get("name"), type=jtype, parent_node=j.find("parent").get("link"),
            child_node=j.find("child").get("link"), pos=(0.0, 0.0, 0.0), axis=axis, limits=limits,
            damping=float(dyn.get("damping", 0.0)) if dyn is not None else 0.0,
            effort_limit=float(limit.get("effort")) if limit is not None and limit.get("effort") else None,
        ))  # fmt: skip
        if dyn is not None and float(dyn.get("friction", 0.0)) > 0:
            ir.warnings.append(f"joint '{j.get('name')}': Coulomb joint friction is not modelled")
    roots = [n for n in links if n not in child_of]
    if len(roots) != 1:
        ir.warnings.append(f"expected one root link, found {len(roots)}: {roots}")
    return ir
