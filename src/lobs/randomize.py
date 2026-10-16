"""Initial-state randomization: distributions + ``sample_initial``.

    sim.reset(seed=3, randomize={"box": {"pos": (Uniform(-1, 1), Uniform(-1, 1), 0.5),
                                          "yaw": Uniform(-3.14, 3.14),
                                          "linvel": Normal((0, 0, 0), 0.1)}})

Sampling uses the run's own seeded RNG, so ``reset(seed=...)`` is reproducible.
"""

from __future__ import annotations

import numpy as np

from . import math3d as m3
from .world import Initial


class Distribution:
    def sample(self, rng: np.random.Generator):
        raise NotImplementedError


class Uniform(Distribution):
    def __init__(self, low, high):
        self.low, self.high = np.asarray(low, float), np.asarray(high, float)

    def sample(self, rng):
        v = rng.uniform(self.low, self.high)
        return float(v) if np.ndim(v) == 0 else tuple(float(x) for x in v)


class Normal(Distribution):
    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, float), np.asarray(std, float)

    def sample(self, rng):
        v = rng.normal(self.mean, self.std)
        return float(v) if np.ndim(v) == 0 else tuple(float(x) for x in v)


class Choice(Distribution):
    def __init__(self, options):
        self.options = list(options)

    def sample(self, rng):
        return sample(self.options[int(rng.integers(len(self.options)))], rng)


def sample(x, rng):
    """Resolve distributions anywhere inside numbers / tuples / lists / dicts."""
    if isinstance(x, Distribution):
        return x.sample(rng)
    if isinstance(x, dict):
        return {k: sample(v, rng) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return tuple(sample(v, rng) for v in x)
    return x


def sample_initial(rng: np.random.Generator, spec: dict) -> dict:
    """``{actor: {pos, quat, yaw, linvel, angvel, q, qd}}`` (each value fixed or a Distribution) ->
    ``{actor: lobs.Initial}``. ``yaw`` is a rotation about +z (composed onto ``quat`` if both are given)."""
    out = {}
    for actor, fields in spec.items():
        fields = dict(fields)
        yaw = fields.pop("yaw", None)
        kw = {k: sample(v, rng) for k, v in fields.items()}
        if yaw is not None:
            q = m3.quat_from_axis_angle([0, 0, 1], float(sample(yaw, rng)))
            base = np.asarray(kw["quat"], float) if kw.get("quat") is not None else m3.IDENTITY_QUAT
            kw["quat"] = tuple(float(x) for x in m3.quat_mul(q, base))
        unknown = set(kw) - {"pos", "quat", "linvel", "angvel", "q", "qd"}
        if unknown:
            raise ValueError(f"randomize[{actor!r}]: unknown fields {sorted(unknown)}")
        out[actor] = Initial(**kw)
    return out
