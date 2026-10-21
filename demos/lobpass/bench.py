"""Throughput gate (L4): aggregate simulated ticks/s of LoBS worlds in separate processes with a random policy.

pixi run lob-bench                                  # (workers x envs-per-worker) grid on stage S0
pixi run lob-bench --workers 14 --per-worker 8 --stage S3c --seconds 20
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from .vec import VecLob


def measure(
    workers: int, per_worker: int, stage: str, seconds: float, seed: int = 0
) -> tuple[float, float]:
    """(aggregate ticks/s, decisions/s) over ``seconds`` of wall time, after a short warm-up."""
    venv = VecLob(workers, per_worker, seed, {stage: 1.0})
    try:
        venv.reset()
        rng = np.random.default_rng(seed)
        n = venv.n

        def act():
            a = rng.uniform(-1, 1, (n, 4)).astype(np.float32)
            a[:, 0] = np.where(rng.random(n) < 0.08, 1.0, -1.0)
            return a

        for _ in range(10):
            venv.step(act())
        t0_ticks = venv.total_ticks()
        t0, steps = time.perf_counter(), 0
        while time.perf_counter() - t0 < seconds:
            venv.step(act())
            steps += n
        wall = time.perf_counter() - t0
        return (venv.total_ticks() - t0_ticks) / wall, steps / wall
    finally:
        venv.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="*", default=[4, 8, 14])
    ap.add_argument("--per-worker", type=int, nargs="*", default=[8])
    ap.add_argument("--stage", default="S0")
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()
    for w in args.workers:
        for k in args.per_worker:
            tps, dps = measure(w, k, args.stage, args.seconds)
            print(
                f"{w:2d} workers x {k:2d} envs, stage {args.stage}: {tps:10,.0f} ticks/s aggregate, {dps:8,.0f} decisions/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
