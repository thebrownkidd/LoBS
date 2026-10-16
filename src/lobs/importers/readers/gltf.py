"""glTF / GLB scenes via trimesh. glTF is meters, Y-up, and carries no physics."""

from __future__ import annotations

from ..ir import IRMaterial, SceneIR, SourceInfo, apply_lobs_annotations, bake_mesh_node


def _node_path(graph, name: str) -> list[str]:
    parents = graph.transforms.parents
    chain, cur = [], name
    while cur is not None and cur != graph.base_frame:
        chain.append(cur)
        cur = parents.get(cur)
    return list(reversed(chain))


def read(path, **_):
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    src = SourceInfo(path=str(path), format="gltf", meters_per_unit=1.0, up_axis="y")
    ir = SceneIR(src)
    seen: dict[str, int] = {}
    for name in sorted(scene.graph.nodes_geometry):
        transform, gname = scene.graph[name]
        mesh = scene.geometry[gname]
        if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
            ir.warnings.append(f"skipped non-mesh geometry '{gname}'")
            continue
        chain = _node_path(scene.graph, name)
        nid = "/".join(chain)
        seen[nid] = seen.get(nid, 0) + 1
        if seen[nid] > 1:
            nid = f"{nid}#{seen[nid]}"
        v = trimesh.transform_points(mesh.vertices, transform)
        node = bake_mesh_node(nid, str(chain[-1]), v, mesh.faces)
        node.tags = [f"group:{c}" for c in chain[:-1]]
        mat = getattr(mesh.visual, "material", None)
        if mat is not None:
            color = getattr(mat, "main_color", None)
            node.material = IRMaterial(
                name=getattr(mat, "name", None),
                color=tuple(float(c) / 255.0 for c in color) if color is not None else None,
                raw={"annotated": False},
            )
        apply_lobs_annotations(node, (mesh.metadata or {}).get("gltf_extras"))
        node.properties["geometry"] = str(gname)
        ir.add(node)
    if not ir.nodes:
        ir.warnings.append("no mesh geometry found")
    return ir
