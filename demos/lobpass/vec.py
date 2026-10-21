"""Batched multi-process vector env.

gymnasium's AsyncVectorEnv pays one pipe round trip per env per decision (about 7 ms on Windows, against ~1 ms of
physics per 10-tick decision), so it tops out near 15k ticks/s here. ``VecLob`` runs ``per_worker`` envs inside each
worker process and sends one message per worker per step, which keeps the cores busy.

Autoreset is "same step": a terminated env is reset immediately and the observation returned is the new
episode's first one; the finished episode's info is returned for that env."""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

from . import constants as C
from .env import LobPassEnv

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


def _worker(conn, ranks, base_seed, stage_mix):
    envs = [LobPassEnv(stage_mix, base_seed, r) for r in ranks]
    k_envs = len(envs)
    last_obs = np.zeros((k_envs, C.OBS_DIM), np.float32)
    finished = [False] * k_envs
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                for k, env in enumerate(envs):
                    opts = data[k] if data is not None else None
                    last_obs[k] = env.reset(options=opts or {})[0]
                    finished[k] = False
                conn.send(last_obs.copy())
            elif cmd == "step":
                actions, autoreset = data
                rew = np.zeros(k_envs, np.float32)
                term = np.zeros(k_envs, bool)
                skipped = np.zeros(k_envs, bool)
                infos = {}
                for k, env in enumerate(envs):
                    if finished[k]:
                        skipped[k] = True
                        continue
                    obs, r, t, _, info = env.step(actions[k])
                    rew[k], term[k], last_obs[k] = r, t, obs
                    if t:
                        infos[k] = info
                        if autoreset:
                            last_obs[k] = env.reset()[0]
                        else:
                            finished[k] = True
                conn.send((last_obs.copy(), rew, term, skipped, infos))
            elif cmd == "mix":
                for env in envs:
                    env.set_stage_mix(data)
                conn.send(True)
            elif cmd == "ticks":
                conn.send(sum(env.total_ticks() for env in envs))
            elif cmd == "close":
                break
    finally:
        for env in envs:
            env.close()
        conn.close()


class VecLob:
    def __init__(
        self, workers: int, per_worker: int = 4, base_seed: int = 0, stage_mix: dict | None = None
    ):
        ctx = mp.get_context("spawn")
        self.workers, self.per_worker = workers, per_worker
        self.n = workers * per_worker
        self.conns, self.procs = [], []
        # one BLAS/OpenMP thread per worker: the workers are the parallelism (restored for the parent afterwards)
        saved = {k: os.environ.get(k) for k in _THREAD_VARS}
        os.environ.update({k: "1" for k in _THREAD_VARS})
        for w in range(workers):
            parent, child = ctx.Pipe()
            ranks = list(range(w * per_worker, (w + 1) * per_worker))
            p = ctx.Process(
                target=_worker,
                args=(child, ranks, base_seed, stage_mix or {"S0": 1.0}),
                daemon=True,
            )
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _slices(self):
        return [slice(w * self.per_worker, (w + 1) * self.per_worker) for w in range(self.workers)]

    def reset(self, options: list | None = None) -> np.ndarray:
        """Reset every env (optionally with a per-env options dict, e.g. ``{"scenario": scn}``)."""
        for conn, sl in zip(self.conns, self._slices(), strict=True):
            conn.send(("reset", None if options is None else options[sl]))
        return np.concatenate([c.recv() for c in self.conns])

    def step(self, actions: np.ndarray, autoreset: bool = True):
        """Returns obs (N,109), reward (N,), terminated (N,), skipped (N,) and a {env index: info} dict."""
        actions = np.asarray(actions, np.float32)
        for conn, sl in zip(self.conns, self._slices(), strict=True):
            conn.send(("step", (actions[sl], autoreset)))
        outs = [c.recv() for c in self.conns]
        obs = np.concatenate([o[0] for o in outs])
        rew = np.concatenate([o[1] for o in outs])
        term = np.concatenate([o[2] for o in outs])
        skipped = np.concatenate([o[3] for o in outs])
        infos = {}
        for w, o in enumerate(outs):
            infos.update({w * self.per_worker + k: v for k, v in o[4].items()})
        return obs, rew, term, skipped, infos

    def set_stage_mix(self, mix: dict) -> None:
        for conn in self.conns:
            conn.send(("mix", mix))
        for conn in self.conns:
            conn.recv()

    def total_ticks(self) -> int:
        for conn in self.conns:
            conn.send(("ticks", None))
        return sum(c.recv() for c in self.conns)

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
