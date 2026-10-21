"""Curriculum: stage mixing, the deterministic evaluation on the fixed sets, and the promotion gate (A10)."""

from __future__ import annotations

import json
import pathlib
from collections import Counter

import numpy as np
import torch

from . import constants as C
from .oracle import eval_seeds
from .ppo import Policy, to_tensor
from .scenario import sample_scenario
from .vec import VecLob


def stage_mix(stage: str) -> dict:
    """75 % current stage, 25 % spread uniformly over earlier stages (all current if there are none)."""
    k = C.STAGES.index(stage)
    if k == 0:
        return {stage: 1.0}
    mix = {s: (1.0 - C.CURRENT_STAGE_MIX) / k for s in C.STAGES[:k]}
    mix[stage] = C.CURRENT_STAGE_MIX
    return mix


def gate_threshold(stage: str, oracle_rate: float | None) -> float:
    base = C.GATES[stage]
    return (
        base if stage in ("S0", "S1") else base * (oracle_rate if oracle_rate is not None else 1.0)
    )


def load_oracle_rate(stage: str, root: str | pathlib.Path = "runs/lobpass") -> float | None:
    p = pathlib.Path(root) / f"oracle_{stage}.json"
    return json.loads(p.read_text())["oracle"] if p.exists() else None


@torch.no_grad()
def evaluate(
    policy: Policy,
    venv: VecLob,
    stage: str,
    n: int = C.EVAL_N,
    seeds: list | None = None,
    keep_infos: bool = False,
) -> dict:
    """Deterministic policy (throw iff p >= 0.5, mean parameters) on the stage's fixed scenarios."""
    seeds = seeds if seeds is not None else eval_seeds(stage, n)
    scns = [sample_scenario(stage, s) for s in seeds]
    outcomes: Counter = Counter()
    rewards, release_t, clamped = [], [], 0
    infos_out: list = []
    for r0 in range(0, len(scns), venv.n):
        chunk = scns[r0 : r0 + venv.n]
        opts = [{"scenario": s} for s in chunk] + [{"scenario": chunk[0]}] * (venv.n - len(chunk))
        obs = venv.reset(opts)
        live = np.array([True] * len(chunk) + [False] * (venv.n - len(chunk)))
        done = ~live.copy()
        while not done.all():
            a, *_ = policy.act(to_tensor(obs), deterministic=True)
            obs, rew, term, skipped, infos = venv.step(a.numpy(), autoreset=False)
            for k, info in infos.items():
                if live[k] and not done[k]:
                    outcomes[info["outcome"]] += 1
                    if keep_infos:
                        infos_out.append(info)
                    rewards.append(float(rew[k]))
                    if info["t_release"] is not None:
                        release_t.append(info["t_release"])
                    clamped += bool(info["clamped"])
                    done[k] = True
    total = max(sum(outcomes.values()), 1)
    return {
        "stage": stage, "n": total, "success": outcomes["DUNK"] / total,
        "outcomes": {k: v / total for k, v in outcomes.items()}, "reward": float(np.mean(rewards)),
        "release_time": float(np.mean(release_t)) if release_t else None, "clamp_rate": clamped / total,
        **({"infos": infos_out} if keep_infos else {}),
    }  # fmt: skip


class Gate:
    """Promotes after two consecutive passing evaluations."""

    def __init__(self, stage: str, oracle_rate: float | None):
        self.threshold = gate_threshold(stage, oracle_rate)
        self.passes = 0

    def update(self, success: float) -> bool:
        self.passes = self.passes + 1 if success >= self.threshold else 0
        return self.passes >= 2
