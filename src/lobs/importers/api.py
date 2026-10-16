"""`import_world`: the one-call importer, plus package writing (plan 10.1 / 11.7 / 11.8)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from .. import __version__
from ..errors import ImportFailure
from ..schema.io import save_world
from ..schema.world import ImportRecord, WorldSpec
from ..validate import required_capabilities
from . import readers
from .decisions import DecisionLog, ImportReport, Overrides, load_profile
from .normalize import normalize
from .physicalize import physicalize

OPTION_KEYS = (
    "units",
    "up_axis",
    "origin",
    "scale",
    "actuate",
    "format",
    "motion",
    "copy_source",
    "reader_opts",
)


def import_world(source, *, motion=None, profile="generic", out=None, overrides=None, units="auto",
                 up_axis="auto", origin="auto", scale=1.0, actuate=None, format=None,
                 copy_source=False, **reader_opts):  # fmt: skip
    """Import a foreign world file. Returns ``(World, ImportReport)`` and writes a `.lobs` package.

    ``motion`` is required (here or in the profile): ``'auto'`` guesses which elements move from
    their class / name / size; ``'dynamic'`` makes everything movable except ground support. Source
    physics data, profile rules and ``overrides`` always take precedence.
    """
    from ..world import World

    source = Path(source)
    prof, prof_digest = load_profile(profile)
    mode = motion or prof.get("motion")
    if mode not in ("auto", "dynamic"):
        raise ImportFailure(
            "pass motion='auto' or motion='dynamic' (no implicit default: it decides what moves)"
        )
    out = Path(out) if out is not None else source.with_suffix(".lobs")
    if out.suffix != ".lobs":
        out = out.with_suffix(".lobs")
    ov = Overrides.load(
        overrides
        if overrides is not None
        else (
            out / "import" / "overrides.yaml"
            if (out / "import" / "overrides.yaml").exists()
            else None
        )
    )

    ir = readers.read(source, format, units=units, **reader_opts)
    log = DecisionLog()
    georef = normalize(ir, log, units=units, up_axis=up_axis, origin=origin, scale=scale)
    if out.exists():
        shutil.rmtree(out / "assets", ignore_errors=True)
    assets = out / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    result = physicalize(
        ir,
        profile=prof,
        motion=mode,
        overrides=ov,
        assets_dir=assets,
        log=log,
        name=source.stem,
        georef=georef,
        actuate=actuate,
    )
    spec, report = result.spec, result.report

    ov_text = (
        (out / "import" / "overrides.yaml").read_text(encoding="utf-8")
        if (out / "import" / "overrides.yaml").exists()
        else None
    )
    spec.imports.append(ImportRecord(source_path=str(source), format=ir.source.format, source_sha256=ir.source.sha256, profile_sha256=prof_digest,
                                     motion_mode=mode, lobs_version=__version__, decisions_path="import/decisions.json",
                                     overrides_sha256=_sha(ov_text) if ov_text else None))  # fmt: skip
    world = World(spec=spec, base_dir=out)
    reg = world.registry()
    report.capabilities_required = sorted(c.value for c in required_capabilities(reg))
    _write_package(out, world, report, prof, mode, source, ir, ov_text, dict(units=units, up_axis=up_axis, origin=origin, scale=scale,
                                                                             actuate=actuate, format=format, motion=mode, reader_opts=reader_opts), copy_source)  # fmt: skip
    return world, report


def update_world(package, overrides=None):
    """Re-run an import with the (edited) overrides stored in the package. Deterministic."""
    package = Path(package)
    rec_file = package / "import" / "options.yaml"
    if not rec_file.exists():
        raise ImportFailure(f"{package} is not an imported world package (no import/options.yaml)")
    opts = yaml.safe_load(rec_file.read_text(encoding="utf-8"))
    spec = yaml.safe_load((package / "world.yaml").read_text(encoding="utf-8"))
    src = spec["imports"][0]["source_path"]
    profile = yaml.safe_load((package / "import" / "profile.lock.yaml").read_text(encoding="utf-8"))
    reader_opts = opts.pop("reader_opts", {}) or {}
    opts.pop("copy_source", None)
    return import_world(
        src, profile=profile, out=package, overrides=overrides, **opts, **reader_opts
    )


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_package(
    out: Path,
    world,
    report: ImportReport,
    prof: dict,
    mode: str,
    source: Path,
    ir,
    ov_text,
    options: dict,
    copy_source: bool,
):
    spec: WorldSpec = world.spec
    save_world(spec, out)
    imp = out / "import"
    imp.mkdir(parents=True, exist_ok=True)
    (imp / "profile.lock.yaml").write_text(yaml.safe_dump(prof, sort_keys=True), encoding="utf-8")
    (imp / "options.yaml").write_text(
        yaml.safe_dump(_plain(options), sort_keys=True), encoding="utf-8"
    )
    (imp / "decisions.json").write_text(report.decisions.to_json(), encoding="utf-8")
    (imp / "report.md").write_text(report.markdown(), encoding="utf-8")
    if ov_text is None:
        review = {}
        for d in report.needs_review:
            review.setdefault(d.target, {})
        template = {"elements": {k: {} for k in sorted(review) if k not in ("scene", "world")}}
        (imp / "overrides.yaml").write_text(
            "# Per-element decisions, keyed by source id. Fields: motion (fixed|dynamic), material, density,\n"
            "# friction, restitution, collision (auto|box_fit|convex_hull|decompose|mesh|none), mass, tags, skip.\n"
            "# Edit, then run:  lobs import --update <this package>\n"
            + yaml.safe_dump(template, sort_keys=True),
            encoding="utf-8",
        )
    if copy_source:
        (out / "source").mkdir(exist_ok=True)
        shutil.copy2(source, out / "source" / source.name)
    card = {
        "title": spec.name, "summary": "", "authors": [], "license": None,
        "source": {"name": source.name, "url": None, "format": ir.source.format, "sha256": ir.source.sha256, "license": None},
        "created_with": {"lobs": __version__, "profile": prof.get("profile"), "motion": mode},
        "tags": [], "counts": report.counts, "georeference": report.georeference,
        "capabilities_required": report.capabilities_required, "validation": {},
    }  # fmt: skip
    (out / "card.yaml").write_text(yaml.safe_dump(card, sort_keys=False), encoding="utf-8")


def _plain(o):
    return json.loads(json.dumps(o, default=str))
