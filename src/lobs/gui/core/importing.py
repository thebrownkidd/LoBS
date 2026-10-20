"""Import helpers for the GUI wizard: run an import, tabulate decisions, build overrides (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ... import import_world

EDITABLE = ("motion", "mass", "density", "friction", "restitution", "collision", "material", "skip")
EXTENSIONS = {
    "glTF / GLB": "*.gltf *.glb", "IFC (BIM)": "*.ifc", "OpenUSD": "*.usd *.usda *.usdc *.usdz",
    "URDF": "*.urdf", "SDF": "*.sdf", "MJCF": "*.xml", "Meshes": "*.obj *.stl *.ply",
}  # fmt: skip


def file_filter() -> str:
    allx = " ".join(EXTENSIONS.values())
    return ";;".join(
        [f"All supported ({allx})"]
        + [f"{k} ({v})" for k, v in EXTENSIONS.items()]
        + ["All files (*)"]
    )


@dataclass
class ImportRequest:
    source: str
    motion: str | None = None  # 'auto' | 'dynamic'; no implicit default
    profile: str = "generic"
    out: str | None = None
    units: str = "auto"
    up_axis: str = "auto"
    scale: float = 1.0
    overrides: dict = field(default_factory=dict)  # {source_id: {field: value}}

    def ready(self) -> bool:
        return bool(self.source) and self.motion in ("auto", "dynamic")


def run_import(req: ImportRequest):
    """Run the library importer. Returns ``(World, ImportReport)``."""
    if not req.ready():
        raise ValueError("choose a file and a motion mode (auto or dynamic) first")
    kw = {
        "motion": req.motion,
        "profile": req.profile,
        "units": req.units,
        "up_axis": req.up_axis,
        "scale": req.scale,
    }
    if req.out:
        kw["out"] = req.out
    kw["overrides"] = (
        {"elements": {k: dict(v) for k, v in req.overrides.items()}} if req.overrides else None
    )
    return import_world(Path(req.source), **kw)


def decision_rows(report, only_review: bool = False, text: str = "") -> list[dict]:
    """Decisions as dicts, optionally only the ones that need a human look and/or matching ``text``."""
    rows = []
    for d in report.decisions.items:
        if only_review and not d.review:
            continue
        row = {"target": d.target, "field": d.field, "value": d.value, "provenance": d.provenance,
               "confidence": d.confidence, "rationale": d.rationale, "review": d.review,
               "editable": d.field.split(".")[0] in EDITABLE}  # fmt: skip
        if text and text.lower() not in " ".join(str(v) for v in row.values()).lower():
            continue
        rows.append(row)
    return rows


def set_override(overrides: dict, target: str, field_: str, value) -> None:
    key = field_.split(".")[0]
    if key not in EDITABLE:
        raise ValueError(f"'{field_}' cannot be overridden (editable: {', '.join(EDITABLE)})")
    overrides.setdefault(target, {})[key] = value


def clear_override(overrides: dict, target: str, field_: str | None = None) -> None:
    if field_ is None:
        overrides.pop(target, None)
    else:
        overrides.get(target, {}).pop(field_.split(".")[0], None)
        if not overrides.get(target):
            overrides.pop(target, None)


def parse_value(field_: str, text: str):
    t = text.strip()
    key = field_.split(".")[0]
    if key in ("mass", "density", "friction", "restitution"):
        return float(t)
    if key == "skip":
        return t.lower() in ("1", "true", "yes", "y")
    return t
