"""Backend registry. Engines are imported lazily; ``import lobs`` never needs them."""

from __future__ import annotations

from importlib import import_module

from ..errors import BackendUnavailable
from .base import PhysicsBackend
from .capabilities import Capability, Support

_MODULES = {
    "mujoco": ("lobs.backends.mujoco_backend", "MujocoBackend", "pip install lobs[mujoco]"),
    "chrono": (
        "lobs.backends.chrono_backend",
        "ChronoBackend",
        "conda install -c conda-forge pychrono (or use the pixi environment)",
    ),
}


def get(name: str) -> type[PhysicsBackend]:
    if name not in _MODULES:
        raise BackendUnavailable(f"unknown backend '{name}'; known: {sorted(_MODULES)}")
    mod, cls, hint = _MODULES[name]
    try:
        backend = getattr(import_module(mod), cls)
    except ImportError as e:
        raise BackendUnavailable(f"backend '{name}' is not installed ({e}). Install: {hint}") from e
    if not backend.available():
        raise BackendUnavailable(f"backend '{name}' is not available. Install: {hint}")
    return backend


def available() -> list[str]:
    out = []
    for name in _MODULES:
        try:
            get(name)
            out.append(name)
        except BackendUnavailable:
            pass
    return out


__all__ = ["Capability", "PhysicsBackend", "Support", "available", "get"]
