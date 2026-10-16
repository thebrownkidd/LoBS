"""`lobs` command line: import / check / info / card."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml


def _origin(text: str):
    if text in ("auto", "keep", "recenter"):
        return text
    try:
        v = [float(x) for x in text.split(",")]
        assert len(v) == 3
        return v
    except Exception as e:  # noqa: BLE001
        raise argparse.ArgumentTypeError("origin must be auto, keep, recenter or x,y,z") from e


def _print_report(report) -> None:
    print(f"imported {report.source}  (profile {report.profile}, motion {report.motion})")
    print("  " + ", ".join(f"{k}={v}" for k, v in report.counts.items()))
    for prov, per in report.decisions.counts().items():
        print(f"  decisions [{prov}]: " + ", ".join(f"{c}={n}" for c, n in per.items()))
    review = report.needs_review
    print(f"  needs review: {len(review)}" + ("  (see import/report.md)" if review else ""))
    for w in report.warnings:
        print(f"  warning: {w}")


def cmd_import(args) -> int:
    from .importers.api import import_world, update_world

    if args.update:
        world, report = update_world(args.update)
        print(f"re-imported {args.update}")
    else:
        if not args.source:
            print("error: give a source file (or --update PACKAGE)", file=sys.stderr)
            return 2
        world, report = import_world(
            args.source, motion=args.motion, profile=args.profile, out=args.out, units=args.units,
            up_axis=args.up_axis, origin=args.origin, scale=args.scale, actuate=args.actuate,
            copy_source=args.copy_source,
        )  # fmt: skip
    _print_report(report)
    print(f"  package: {world.base_dir}")
    return 0


def cmd_check(args) -> int:
    from .check import check_world
    from .world import load_world

    world = load_world(args.world)
    res = check_world(world, args.backend, args.seconds)
    print(res.summary())
    card = Path(args.world) / "card.yaml"
    if card.exists():
        data = yaml.safe_load(card.read_text(encoding="utf-8")) or {}
        data.setdefault("validation", {})[args.backend] = {
            "engine_version": res.engine_version, "passed": res.passed,
            "realtime_factor": round(res.realtime_factor, 2), "seconds": res.seconds,
            "date": datetime.now(UTC).date().isoformat(),
        }  # fmt: skip
        card.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return 0 if res.passed else 1


def cmd_info(args) -> int:
    from .validate import required_capabilities
    from .world import load_world

    w = load_world(args.world)
    reg = w.registry()
    print(f"{w.spec.name}: {len(reg.actors)} actors ({len(reg.dynamic_order)} dynamic bodies, "
          f"{len(reg.bodies) - len(reg.dynamic_order)} static), {len(reg.joints)} joints, "
          f"{len(reg.anchors)} anchors, {len(w.spec.measures)} measures")  # fmt: skip
    print("requires:", ", ".join(sorted(c.value for c in required_capabilities(reg))))
    for imp in w.spec.imports:
        print(
            f"imported from {imp.source_path} ({imp.format}, motion={imp.motion_mode}, lobs {imp.lobs_version})"
        )
    if w.spec.georeference:
        print("georeference:", w.spec.georeference.model_dump())
    return 0


def _preview(world_path) -> Path:
    from .viz import write_png
    from .world import load_world

    world = load_world(world_path)
    sim = world.simulate("mujoco", record=False)
    try:
        img = sim.render(960, 540)
    finally:
        sim.close()
    return write_png(Path(world_path) / "previews" / "overview.png", img)


def cmd_schema(args) -> int:
    import json

    from .schema.io import json_schema

    print(json.dumps(json_schema(), indent=2))
    return 0


def cmd_card(args) -> int:
    if args.preview:
        print(f"preview written to {_preview(args.world)}")
    card = Path(args.world) / "card.yaml"
    print(card.read_text(encoding="utf-8") if card.exists() else "(no card.yaml in this world)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="lobs", description="The Laws of Bullshit: world tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("import", help="import a foreign world file into a .lobs package")
    pi.add_argument("source", nargs="?")
    pi.add_argument("-o", "--out")
    pi.add_argument(
        "--update", metavar="PACKAGE", help="re-run an import using the edited overrides"
    )
    pi.add_argument("--profile", default="generic")
    pi.add_argument(
        "--motion", choices=["auto", "dynamic"], help="what moves when the source does not say"
    )
    pi.add_argument("--units", default="auto")
    pi.add_argument("--up-axis", default="auto", dest="up_axis")
    pi.add_argument("--origin", type=_origin, default="auto")
    pi.add_argument("--scale", type=float, default=1.0)
    pi.add_argument("--actuate", default=None)
    pi.add_argument("--copy-source", action="store_true")
    pi.set_defaults(fn=cmd_import)
    pc = sub.add_parser("check", help="headless stability check")
    pc.add_argument("world")
    pc.add_argument("--backend", default="mujoco")
    pc.add_argument("--seconds", type=float, default=2.0)
    pc.set_defaults(fn=cmd_check)
    pn = sub.add_parser("info", help="summarize a world")
    pn.add_argument("world")
    pn.set_defaults(fn=cmd_info)
    ps = sub.add_parser("schema", help="print the JSON Schema of world files (for editors / GUIs)")
    ps.set_defaults(fn=cmd_schema)
    pk = sub.add_parser("card", help="print a world card")
    pk.add_argument("world")
    pk.add_argument("--preview", action="store_true", help="render previews/overview.png")
    pk.set_defaults(fn=cmd_card)
    args = p.parse_args(argv)
    try:
        return args.fn(args)
    except Exception as e:  # noqa: BLE001 - CLI boundary: report cleanly
        from .errors import LobsError

        if isinstance(e, LobsError):
            print(f"error: {e}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
