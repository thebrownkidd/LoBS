"""User-registered custom measure functions. Only the *name* is serialized in worlds."""

from __future__ import annotations

FUNCTIONS: dict = {}


def register_function(name: str):
    def deco(fn):
        FUNCTIONS[name] = fn
        return fn

    return deco
