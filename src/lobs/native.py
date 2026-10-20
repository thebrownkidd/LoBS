"""Optional native tick kernel (src/native). ``LOBS_NATIVE=0`` forces the pure-Python reference path."""

from __future__ import annotations

import os

_mod = None
_tried = False


def module():
    """The compiled ``lobs._native`` module, or None when it is not built / disabled / incompatible."""
    global _mod, _tried
    if _tried:
        return _mod
    _tried = True
    if os.environ.get("LOBS_NATIVE", "1") == "0":
        return None
    try:
        import mujoco  # noqa: F401  (the kernel uses the MuJoCo library this package loaded)

        from . import _native

        _native.init_mujoco()
        _mod = _native
    except Exception:  # noqa: BLE001 - not built or ABI mismatch: fall back silently
        _mod = None
    return _mod


def available() -> bool:
    return module() is not None
