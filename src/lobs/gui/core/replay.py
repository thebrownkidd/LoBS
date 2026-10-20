"""Replay of a recorded run: frames, evaluation of new measures, timeline lanes (no Qt)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...registry import build_registry
from ...trajectory import Trajectory
from . import exprs


@dataclass
class Lane:
    label: str
    kind: str  # 'state' (integer state index with names) | 'level' (non-negative magnitude)
    t: np.ndarray
    v: np.ndarray
    names: list = field(default_factory=list)


class Replay:
    def __init__(self, traj: Trajectory):
        self.traj = traj
        self.spec = traj.spec
        self.reg = build_registry(traj.spec, traj.base_dir)
        self.frames = list(traj.frames())
        self.index = 0
        self.loop = True

    @classmethod
    def load(cls, path) -> Replay:
        return cls(Trajectory.load(path))

    @classmethod
    def from_sim(cls, sim) -> Replay:
        return cls(sim.trajectory())

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def times(self) -> np.ndarray:
        return self.traj.t

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    def seek_time(self, t: float) -> int:
        self.index = int(np.clip(np.searchsorted(self.times, t), 0, len(self) - 1))
        return self.index

    def frame(self, i: int | None = None):
        return self.frames[self.index if i is None else i]

    def advance(self, dt: float) -> bool:
        """Move forward by ``dt`` seconds of recorded time. Returns False at the end (when not looping)."""
        t = self.times[self.index] + dt
        if t > self.duration:
            if not self.loop:
                self.index = len(self) - 1
                return False
            t = t - self.duration
        self.seek_time(t)
        return True

    # ------------------------------------------------------------ evaluation
    def evaluate(self, text: str) -> np.ndarray:
        """Evaluate a measure expression over the whole recording (library errors propagate unchanged)."""
        m = exprs.evaluate(text, self.traj.world)
        return self.traj.evaluate(m)

    def recorded_measures(self) -> list[str]:
        return list(self.traj.measures)

    # ------------------------------------------------------------ timeline
    def forces_at(self, i: int | None = None) -> dict:
        i = self.index if i is None else i
        return {k: v[i] for k, v in self.traj.forces.items()}

    def labels_at(self, i: int | None = None) -> list:
        """``[(actor_path, 'behavior: state')]`` for the recorded behavior states at a frame."""
        i = self.index if i is None else i
        names = self.traj.meta.get("behavior_states", {})
        out = []
        for b in self.spec.behaviors:
            if b.on is None or b.name not in self.traj.behaviors:
                continue
            idx = int(self.traj.behaviors[b.name][i])
            st = names.get(b.name, [])
            out.append((b.on, f"{b.name}: {st[idx] if idx < len(st) else idx}"))
        return out

    def lanes(self) -> list[Lane]:
        return lanes_from(
            self.times,
            self.traj.behaviors,
            self.traj.meta.get("behavior_states", {}),
            self.traj.forces,
        )


def lanes_from(t, behaviors: dict, state_names: dict, forces: dict) -> list[Lane]:
    out = []
    for name, idx in behaviors.items():
        out.append(
            Lane(
                f"behavior {name}",
                "state",
                np.asarray(t),
                np.asarray(idx),
                list(state_names.get(name, [])),
            )
        )
    for name, f in forces.items():
        out.append(
            Lane(f"force {name}", "level", np.asarray(t), np.linalg.norm(np.asarray(f), axis=1))
        )
    return out


def lanes_from_sim(sim) -> list[Lane]:
    t = sim.history.time
    behaviors, forces = {}, {}
    if sim._beh_hist and len(sim._beh_hist) == len(t):
        behaviors = {k: np.array([h[k] for h in sim._beh_hist]) for k in sim._beh_hist[0]}
    if sim._force_hist and len(sim._force_hist) == len(t):
        forces = {k: np.stack([h[k] for h in sim._force_hist]) for k in sim._force_hist[0]}
    names = {b.spec.name: b.state_names for b in sim._beh.items} if sim._beh is not None else {}
    return lanes_from(t, behaviors, names, forces)


def segments(lane: Lane):
    """Run-length encode a state lane: ``[(t0, t1, state_index)]``."""
    if lane.kind != "state" or len(lane.t) == 0:
        return []
    out, start = [], 0
    for i in range(1, len(lane.v) + 1):
        if i == len(lane.v) or lane.v[i] != lane.v[start]:
            out.append(
                (float(lane.t[start]), float(lane.t[min(i, len(lane.t) - 1)]), int(lane.v[start]))
            )
            start = i
    return out
