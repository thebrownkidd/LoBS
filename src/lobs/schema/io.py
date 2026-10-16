"""World (de)serialization: YAML/JSON single files and `.lobs` package directories."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from ..errors import LobsError
from .world import SCHEMA_VERSION, WorldSpec

PACKAGE_WORLD_FILE = "world.yaml"


def migrate(data: dict) -> dict:
    """Upgrade an older world dict to the current schema version."""
    v = data.get("lobs_schema", 1)
    if v > SCHEMA_VERSION:
        raise LobsError(f"world file schema {v} is newer than supported ({SCHEMA_VERSION})")
    return data


def _dump(spec: WorldSpec, path: Path) -> None:
    data = spec.model_dump(mode="json")
    if path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def save_world(spec: WorldSpec, path) -> None:
    path = Path(path)
    if path.suffix == ".lobs":
        path.mkdir(parents=True, exist_ok=True)
        _dump(spec, path / PACKAGE_WORLD_FILE)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _dump(spec, path)


def load_world(path) -> WorldSpec:
    path = Path(path)
    if path.is_dir():
        path = path / PACKAGE_WORLD_FILE
    if not path.exists():
        raise LobsError(f"world file not found: {path}")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    return WorldSpec.model_validate(migrate(data))


def json_schema() -> dict:
    return WorldSpec.model_json_schema()
