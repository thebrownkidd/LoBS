"""IFC (BIM) reader via IfcOpenShell.

Geometry is taken from IfcOpenShell's tessellation in world coordinates of the *project* coordinate
system and in project length units (the unit scale is reported to normalization). Elements keep their
GlobalId as identity, their IFC class as semantic class, and their spatial containment as tags.
"""

from __future__ import annotations

import numpy as np

from ..ir import IRMaterial, IRPhysics, SceneIR, SourceInfo, apply_lobs_annotations, bake_mesh_node

DEFAULT_EXCLUDE = ("IfcOpeningElement", "IfcSpace", "IfcAnnotation", "IfcGrid", "IfcVirtualElement")
_WEIGHT_KEYS = ("NetWeight", "GrossWeight")


def _ancestors(ifc_el, util_element):
    """Names of the storey / building / site containing an element (by class)."""
    out: dict[str, str] = {}
    cur = util_element.get_container(ifc_el) or util_element.get_aggregate(ifc_el)
    seen = 0
    while cur is not None and seen < 12:
        seen += 1
        for cls, key in (
            ("IfcBuildingStorey", "storey"),
            ("IfcBuilding", "building"),
            ("IfcSite", "site"),
        ):
            if cur.is_a(cls) and key not in out and getattr(cur, "Name", None):
                out[key] = str(cur.Name)
        cur = util_element.get_aggregate(cur) or util_element.get_container(cur)
    return out


def _georef(ifc):
    maps = ifc.by_type("IfcMapConversion")
    if not maps:
        return None
    mc = maps[0]
    crs = getattr(mc, "TargetCRS", None)
    name = getattr(crs, "Name", None) if crs is not None else None
    return {
        "crs": str(name) if name else None,
        "offset": (float(mc.Eastings), float(mc.Northings), float(mc.OrthogonalHeight or 0.0)),
        "meters": True,  # map units are assumed to be metres
    }


def read(path, *, exclude=DEFAULT_EXCLUDE, **_):
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element as util_element
    import ifcopenshell.util.unit as util_unit

    ifc = ifcopenshell.open(str(path))
    scale = float(util_unit.calculate_unit_scale(ifc))
    src = SourceInfo(
        path=str(path), format="ifc", meters_per_unit=scale, up_axis="z", georef=_georef(ifc)
    )
    src.notes.append(f"IFC schema {ifc.schema}; project length unit = {scale:g} m")
    ir = SceneIR(src)
    products = [
        p
        for p in ifc.by_type("IfcProduct")
        if p.Representation and not any(p.is_a(c) for c in exclude)
    ]
    if not products:
        ir.warnings.append("no geometric products found")
        return ir
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(
        settings, ifc, 1, include=products
    )  # 1 thread: deterministic order
    done: set[str] = set()
    if it.initialize():
        while True:
            shape = it.get()
            el = ifc.by_id(shape.id)
            verts = np.asarray(shape.geometry.verts, dtype=float).reshape(-1, 3)
            faces = np.asarray(shape.geometry.faces, dtype=int).reshape(-1, 3)
            if len(verts) and len(faces):
                node = bake_mesh_node(
                    el.GlobalId,
                    str(el.Name or f"{el.is_a()}_{el.id()}"),
                    verts,
                    faces,
                    semantic_class=el.is_a(),
                )
                _decorate(node, el, util_element)
                ir.add(node)
                done.add(el.GlobalId)
            if not it.next():
                break
    missing = [p for p in products if p.GlobalId not in done]
    if missing:
        ir.warnings.append(
            f"{len(missing)} products had no tessellated geometry (e.g. {missing[0].is_a()} '{missing[0].Name}')"
        )
    return ir


def _decorate(node, el, util_element) -> None:
    for key, val in _ancestors(el, util_element).items():
        node.tags.append(f"{key}:{val}")
    typ = util_element.get_type(el)
    if typ is not None and getattr(typ, "Name", None):
        node.tags.append(f"type:{typ.Name}")
    psets = util_element.get_psets(el)
    node.properties["psets"] = {
        k: {a: b for a, b in v.items() if a != "id"} for k, v in psets.items()
    }
    # material
    mats = [m for m in util_element.get_materials(el) if getattr(m, "Name", None)]
    if mats:
        density = None
        for m in mats:
            d = util_element.get_psets(m).get("Pset_MaterialCommon", {}).get("MassDensity")
            if d:
                density = float(d)
                break
        node.material = IRMaterial(
            name=" + ".join(str(m.Name) for m in mats), density=density, raw={"annotated": False}
        )
    # stated weight
    for pname, props in psets.items():
        if pname.startswith(("Qto_", "BaseQuantities")):
            for k in _WEIGHT_KEYS:
                if props.get(k):
                    node.physics = node.physics or IRPhysics()
                    node.physics.mass = float(props[k])
                    break
    # annotations authored in the BIM tool: property set 'Pset_LoBS'
    own = psets.get("Pset_LoBS")
    if own:
        apply_lobs_annotations(node, {f"lobs:{k}": v for k, v in own.items() if k != "id"})
