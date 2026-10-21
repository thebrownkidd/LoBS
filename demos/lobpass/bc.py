"""Behaviour cloning from the oracle, to warm-start PPO.

The oracle finds a dunking release for (almost) every scenario. Each solved scenario gives one demonstration: the
observations at every decision before the release (label: wait) and at the release decision (label: throw, with the
oracle's action). Pure PPO learns the timing and the throw parameters slowly because the success region is tiny;
cloning puts the policy near it, and PPO then fine-tunes against the real environment.

    pixi run -e lob python -m demos.lobpass.bc --stages S0 S1 --n 4000 --run-id main
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

from . import constants as C

DEMO_SEED = 1_000_000  # training scenarios; the evaluation sets start at C.EVAL_SEED
_ENV = None


def _demo(args):
    """One scenario -> (observations (K+1, 109), action (4,)) or None when the oracle found nothing."""
    global _ENV
    from . import oracle
    from .env import LobPassEnv
    from .scenario import sample_scenario

    stage, seed = args
    if _ENV is None:
        _ENV = LobPassEnv({stage: 1.0})
    scn = sample_scenario(stage, seed)
    sol = oracle.solve(_ENV, scn)
    if not sol["success"]:
        return None
    k_rel = (sol["release"] - C.WINDUP_TICKS) // C.DECISION_TICKS
    obs, _ = _ENV.reset(options={"scenario": scn})
    rows = [obs]
    for _ in range(k_rel):
        obs = _ENV.step(np.array([-1.0, 0.0, 0.0, 0.0]))[0]
        rows.append(obs)
    return np.array(rows, np.float32), np.asarray(sol["action"], np.float32)


def build_dataset(spec: dict, workers: int = 12, offset: int = 0) -> dict:
    """``spec`` maps stage -> number of scenarios to try (some, e.g. in S3c, have no dunking release)."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(st, DEMO_SEED + 100_000 * C.STAGES.index(st) + offset + i) for st, n in spec.items() for i in range(n)]
    with ProcessPoolExecutor(workers) as ex:
        out = [r for r in ex.map(_demo, jobs, chunksize=8) if r is not None]
    obs, throw, act = [], [], []
    for rows, a in out:
        obs.append(rows)
        throw.append(np.r_[np.zeros(len(rows) - 1), 1.0])
        act.append(np.tile(a[1:], (len(rows), 1)))
    return {"obs": np.concatenate(obs), "throw": np.concatenate(throw).astype(np.float32),
            "act": np.concatenate(act).astype(np.float32), "n_scenarios": len(out), "n_jobs": len(jobs)}  # fmt: skip


def fit(policy, data: dict, epochs: int = 40, batch: int = 512, lr: float = 1e-3, pos_weight: float = 6.0, seed: int = 0):
    """Supervised fit of the actor: weighted BCE on the throw logit, MSE on the parameter means at throws."""
    import torch
    import torch.nn.functional as F

    g = torch.Generator().manual_seed(seed)
    obs = torch.as_tensor(data["obs"])
    throw = torch.as_tensor(data["throw"])
    act = torch.as_tensor(data["act"])
    n = len(obs)
    opt = torch.optim.Adam(policy.actor.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * (n // batch + 1))
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot_t = tot_a = 0.0
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            out = policy.actor(obs[idx])
            logit, mean = out[:, 0], out[:, 1:]
            lt = F.binary_cross_entropy_with_logits(logit, throw[idx], pos_weight=torch.tensor(pos_weight))
            m = throw[idx] > 0.5
            la = F.mse_loss(mean[m], act[idx][m]) if m.any() else logit.sum() * 0
            loss = lt + 10.0 * la
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot_t += lt.item() * len(idx)
            tot_a += la.item() * len(idx)
        hist.append((tot_t / n, tot_a / n))
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  bc epoch {ep:3d}: throw loss {hist[-1][0]:.4f}  action mse {hist[-1][1]:.5f}", flush=True)
    return hist


def main() -> None:
    import torch

    from . import curriculum as cur
    from .ppo import Policy
    from .train import ROOT, save_ckpt
    from .vec import VecLob

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", nargs="+", default=["S0:3000", "S1:3000"], help="stage:count ... scenarios to try per stage")
    ap.add_argument("--run-id", default="bc")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default=None, help="start from this checkpoint instead of a fresh policy")
    ap.add_argument("--log-std", type=float, default=-1.3, help="exploration std (log) PPO starts with afterwards")
    ap.add_argument("--out", default="ckpt_bc.pt")
    ap.add_argument("--eval-stages", nargs="*", default=None, help="evaluate on these (default: the spec's stages)")
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()
    run_dir = ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    spec = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in args.spec}
    stages = list(spec)
    data = build_dataset(spec, args.workers)
    print(f"dataset: {data['n_scenarios']}/{data['n_jobs']} scenarios solved, {len(data['obs'])} decisions "
          f"({data['throw'].sum():.0f} throws) in {time.time() - t0:.0f}s", flush=True)  # fmt: skip
    torch.manual_seed(0)
    policy = Policy()
    if args.init:
        from .train import load_policy

        policy = load_policy(args.init)
    fit(policy, data, epochs=args.epochs, lr=args.lr)
    with torch.no_grad():
        policy.log_std.fill_(args.log_std)
    cfg = {"ppo": {"hidden": policy.actor[0].out_features}, "bc": {"spec": spec}}
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4, eps=1e-5)
    save_ckpt(run_dir / args.out, policy, opt, stages[0], 0, 0, 0, cfg)
    print(f"saved {run_dir / args.out}", flush=True)
    if not args.no_eval:
        venv = VecLob(12, 4, 0, {stages[0]: 1.0})
        try:
            for st in args.eval_stages or stages:
                ev = cur.evaluate(policy, venv, st)
                print(f"  eval {st}: DUNK {ev['success']:.3f} {({k: round(v, 3) for k, v in ev['outcomes'].items()})}", flush=True)
        finally:
            venv.close()
    _ = pathlib.Path


if __name__ == "__main__":
    main()
