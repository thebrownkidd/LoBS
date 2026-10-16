"""Content-addressed mesh assets. Plain OBJ (v / f only) so backends need no mesh library."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from .errors import LobsError


def mesh_bytes(vertices, faces) -> bytes:
    v = np.asarray(vertices, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=int).reshape(-1, 3)
    lines = [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    return ("\n".join(lines) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def store_mesh(assets_dir: Path, vertices, faces, rel_prefix: str = "assets") -> str:
    """Write a mesh into ``assets_dir`` named by its content hash; returns the package-relative path."""
    data = mesh_bytes(vertices, faces)
    name = f"{sha256_bytes(data)[:24]}.obj"
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    path = assets_dir / name
    if not path.exists():
        path.write_bytes(data)
    return f"{rel_prefix}/{name}"


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts, faces = [], []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):  # fan-triangulate polygons
                faces.append([idx[0], idx[k], idx[k + 1]])
    return np.array(verts, dtype=float).reshape(-1, 3), np.array(faces, dtype=int).reshape(-1, 3)


@lru_cache(maxsize=512)
def _load_cached(path: str, mtime: float):
    return read_obj(Path(path))


def load_mesh(base_dir, rel: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a mesh asset referenced by a world file (relative to ``base_dir``)."""
    if base_dir is None:
        raise LobsError(
            f"mesh '{rel}' is referenced but the world has no asset directory (load it from a file)"
        )
    path = (Path(base_dir) / rel).resolve()
    if not path.exists():
        raise LobsError(f"mesh asset not found: {path}")
    return _load_cached(str(path), path.stat().st_mtime)
