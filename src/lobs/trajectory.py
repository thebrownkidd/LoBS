"""Trajectory: recorded states, measures and metadata; save/load as NPZ; replay of new measures."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import __version__
from .errors import LobsError
from .measure import build_evaluator
from .measure.nodes import as_measure
from .registry import build_registry
from .schema import WorldSpec
from .schema.world import SCHEMA_VERSION
from .state import ContactSet, StateFrame

_FRAME_FIELDS = (
    "body_pos",
    "body_quat",
    "body_linvel",
    "body_angvel",
    "joint_q",
    "joint_qd",
    "joint_effort",
)


class Trajectory:
    def __init__(self, spec: WorldSpec, t: np.ndarray, arrays: dict, measures: dict,
                 actions: np.ndarray, contacts: dict | None, meta: dict):  # fmt: skip
        self.spec, self.t, self.arrays, self.measures = spec, t, arrays, measures
        self.actions, self.contacts, self.meta = actions, contacts, meta

    # ------------------------------------------------------------ build
    @classmethod
    def from_sim(cls, sim, measures=None) -> Trajectory:
        names = sim.history.names() if measures is None else list(measures)
        frames = sim._frames
        if not frames:
            raise LobsError("nothing recorded (simulate with record=True)")
        arrays = {f: np.stack([getattr(fr, f) for fr in frames]) for f in _FRAME_FIELDS}
        contacts = None
        if frames[0].contacts is not None:
            cs = [fr.contacts for fr in frames]
            contacts = {
                "tick": np.concatenate([np.full(len(c), i) for i, c in enumerate(cs)]).astype(int),
                **{
                    k: np.concatenate([getattr(c, k) for c in cs])
                    for k in (
                        "body_a",
                        "body_b",
                        "point",
                        "normal",
                        "depth",
                        "force_on_b",
                        "normal_force",
                    )
                },
            }
        meas = {n: sim.history[n] for n in names}
        meta = {
            "lobs_version": __version__,
            "schema": SCHEMA_VERSION,
            "backend": sim.info,
            "seed": sim.seed,
            "dt": sim.spec.timing.dt,
            "substeps": sim.spec.timing.substeps,
            "record_every": sim.record_every,
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "created": datetime.now(UTC).isoformat(),
            "body_order": [sim.reg.bodies[i].path for i in sim.reg.dynamic_order],
            "joint_order": [sim.reg.joints[i].path for i in sim.reg.dof_joints],
            "actuator_order": sim.action_names,
            "measures": {k: v for k, v in sim.evaluator.meta().items() if k in meas},
        }
        return cls(sim.spec.model_copy(deep=True), sim.history.time, arrays, meas,
                   np.stack(sim._actions) if sim._actions else np.zeros((len(frames), 0)),
                   contacts, meta)  # fmt: skip

    # ------------------------------------------------------------ io
    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = {"meta": np.array(json.dumps(self.meta, default=str)),
               "world": np.array(self.spec.model_dump_json()), "t": self.t, "actions": self.actions}  # fmt: skip
        out.update(self.arrays)
        if self.contacts is not None:
            out.update({f"contact_{k}": v for k, v in self.contacts.items()})
        out.update({f"m/{k}": v for k, v in self.measures.items()})
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **out)
        return path

    @classmethod
    def load(cls, path) -> Trajectory:
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        spec = WorldSpec.model_validate_json(str(z["world"]))
        arrays = {f: z[f] for f in _FRAME_FIELDS}
        contacts = None
        if "contact_tick" in z.files:
            contacts = {k[len("contact_") :]: z[k] for k in z.files if k.startswith("contact_")}
        meas = {k[2:]: z[k] for k in z.files if k.startswith("m/")}
        return cls(spec, z["t"], arrays, meas, z["actions"], contacts, meta)

    # ------------------------------------------------------------ replay
    @property
    def world(self):
        """A World bound to this trajectory's spec: add anchors/measures, then ``evaluate``."""
        from .world import World

        return World(spec=self.spec)

    def __len__(self):
        return len(self.t)

    def frames(self):
        n_contacts = None
        if self.contacts is not None:
            n_contacts = self.contacts["tick"]
        for i in range(len(self.t)):
            cs = None
            if self.contacts is not None:
                m = n_contacts == i
                cs = ContactSet(**{k: v[m] for k, v in self.contacts.items() if k != "tick"})
            yield StateFrame(
                t=float(self.t[i]), **{f: self.arrays[f][i] for f in _FRAME_FIELDS}, contacts=cs
            )

    def evaluate(self, measure) -> np.ndarray:
        """Evaluate a Measure (or a measure name from the world) over the recorded frames.

        Stateful measures (rates via ``derivative``, integrals, windows, ...) match a live run
        only when the trajectory was recorded with ``record_every=1``.
        """
        reg = build_registry(self.spec)
        if isinstance(measure, str):
            ev = build_evaluator(reg, self.spec.measures, only=[measure])
            key = measure
        else:
            ev = build_evaluator(reg, self.spec.measures, {"__adhoc__": as_measure(measure).node},
                                 only=["__adhoc__"])  # fmt: skip
            key = "__adhoc__"
        if "contacts" in ev.needs and self.contacts is None:
            raise LobsError(
                "this measure needs contacts but the trajectory has none (record_contacts=True)"
            )
        if "accel" in ev.needs:
            raise LobsError("accelerations are not recorded; cannot replay acceleration measures")
        ev.reset()
        return np.asarray([ev.evaluate(f)[key] for f in self.frames()])

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as e:
            raise LobsError("to_pandas needs pandas (pip install pandas)") from e
        cols = {"t": self.t}
        for k, v in self.measures.items():
            if v.ndim == 1:
                cols[k] = v
            else:
                flat = v.reshape(len(v), -1)
                for j in range(flat.shape[1]):
                    cols[f"{k}[{j}]"] = flat[:, j]
        return pd.DataFrame(cols)
