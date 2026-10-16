"""Import a foreign world (here a glTF a designer might export), review the decisions, simulate it.

The importer never silently decides physics: every choice is logged with where it came from and how
confident it is, guesses are flagged for review, and you can override any of them by source id.
"""

import tempfile
from pathlib import Path

import numpy as np
import trimesh

import lobs
from lobs import props

work = Path(tempfile.mkdtemp())

# --- stand-in for a file made in another tool: a floor, a crate on it, and a wall (glTF is Y-up)
scene = trimesh.Scene()
for name, extents, centre in (
    ("floor", (10, 0.2, 10), (0, -0.1, 0)),
    ("wooden_crate", (0.4, 0.4, 0.4), (1.0, 0.2, 0.5)),
    ("north_wall", (10, 3, 0.2), (0, 1.5, -5)),
):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(centre)
    scene.add_geometry(mesh, node_name=name, geom_name=name)
(work / "room.glb").write_bytes(scene.export(file_type="glb"))

# --- import. motion= is required: "auto" guesses what moves from class / name / size, "dynamic" lets
# everything move except ground support. Source data, profile rules and overrides always win.
world, report = lobs.import_world(work / "room.glb", motion="auto", out=work / "room.lobs")
print(f"imported {report.counts}")
for d in report.needs_review:  # the guesses worth a human look
    print(f"  review: {d.target:14} {d.field:9} = {d.value!s:10} ({d.provenance}, {d.confidence}) {d.rationale}")

# --- the world is ordinary: add measures, simulate on either engine
crate = world.actor("wooden_crate")
world.measure("crate_height", props.height(crate))
check = lobs.check_world(world, "mujoco", seconds=1.5)
print(check.summary().splitlines()[0])

sim = world.simulate("mujoco")
sim.run(seconds=1.5)
print(f"crate rests at {sim['crate_height']:.3f} m (a 0.4 m cube on the floor: 0.2)")

# --- the saved package (world.yaml, assets/, import/decisions.json, report.md, overrides.yaml, card.yaml)
print("package contents:", sorted(p.name for p in (work / "room.lobs").iterdir()))

# --- change a decision without touching the file: edit overrides, then re-import deterministically
(work / "room.lobs" / "import" / "overrides.yaml").write_text("elements:\n  wooden_crate: {motion: fixed}\n")
world2, _ = lobs.importers.api.update_world(work / "room.lobs")
crate2 = next(a for a in world2.spec.actors if a.name == "wooden_crate")
print("after the override the crate is", crate2.root, "at", np.asarray(crate2.pose.pos).round(2))
