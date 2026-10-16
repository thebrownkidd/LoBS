"""Readers: one per source format. ``read`` dispatches by extension and imports lazily."""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path

from ...errors import ImporterUnavailable, ImportFailure

_BY_EXT = {
    ".glb": "gltf", ".gltf": "gltf",
    ".obj": "mesh", ".stl": "mesh", ".ply": "mesh", ".off": "mesh",
    ".ifc": "ifc",
    ".usd": "usd", ".usda": "usd", ".usdc": "usd", ".usdz": "usd",
    ".urdf": "urdf",
    ".sdf": "sdf", ".world": "sdf",
    ".xml": "mjcf", ".mjcf": "mjcf",
}  # fmt: skip
_HINT = {
    "gltf": "pip install lobs[gltf]",
    "mesh": "pip install lobs[mesh]",
    "ifc": "pip install lobs[ifc]",
    "usd": "pip install lobs[usd]",
    "urdf": "pip install lobs[robotics]",
    "sdf": "(no extra needed)",
    "mjcf": "pip install lobs[mujoco]",
}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_format(path) -> str:
    ext = Path(path).suffix.lower()
    if ext not in _BY_EXT:
        raise ImportFailure(
            f"cannot import '{path}': unknown extension '{ext}' (known: {sorted(_BY_EXT)})"
        )
    return _BY_EXT[ext]


def read(path, format: str | None = None, **opts):  # noqa: A002
    """Read ``path`` into a SceneIR."""
    path = Path(path)
    if not path.exists():
        raise ImportFailure(f"source not found: {path}")
    fmt = format or detect_format(path)
    try:
        mod = import_module(f"lobs.importers.readers.{fmt}")
    except ImportError as e:
        raise ImporterUnavailable(
            f"the '{fmt}' importer needs a missing dependency ({e}). {_HINT.get(fmt, '')}"
        ) from e
    ir = mod.read(path, **opts)
    ir.source.sha256 = sha256_file(path)
    return ir
