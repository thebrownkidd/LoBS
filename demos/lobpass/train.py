"""Curriculum PPO training.

pixi run -e lob lob-train --stages S0,S1 --run-id first
pixi run -e lob lob-train --stages S2 --run-id first --resume      # continue the same run at a later stage
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import Counter, deque

import torch
import yaml

from . import constants as C
from . import curriculum as cur
from . import obs as O
from .ppo import Policy, Rollout, gae, to_tensor, update
from .vec import VecLob

ROOT = pathlib.Path("runs/lobpass")


def save_ckpt(path, policy, opt, stage, total, stage_decisions, passes, cfg):
    torch.save(
        {
            "policy": policy.state_dict(),
            "opt": opt.state_dict(),
            "stage": stage,
            "decisions": total,
            "stage_decisions": stage_decisions,
            "passes": passes,
            "cfg": cfg,
            "obs_version": O.OBS_VERSION,
        },
        path,
    )


def load_policy(path, hidden: int = C.HIDDEN) -> Policy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert ck["obs_version"] == O.OBS_VERSION, (
        "checkpoint was trained with a different observation layout"
    )
    p = Policy(hidden=ck["cfg"]["ppo"].get("hidden", hidden))
    p.load_state_dict(ck["policy"])
    return p


def log(fp, row):
    fp.write(json.dumps(row) + "\n")
    fp.flush()


def run(args) -> int:
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    run_dir = ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(cfg["seed"])
    workers, per = args.workers or cfg["workers"], args.per_worker or cfg["per_worker"]
    n_envs = workers * per
    n_steps = max(cfg["batch_steps"] // n_envs, 8)
    pp = cfg["ppo"]
    policy = Policy(hidden=pp["hidden"])
    opt = torch.optim.Adam(policy.parameters(), lr=pp["lr"], eps=1e-5)
    total, stages = 0, args.stages.split(",")
    resume_stage, resume_sd, passes0 = None, 0, 0
    latest = run_dir / "latest.pt"
    if args.resume and latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["policy"])
        opt.load_state_dict(ck["opt"])
        total, resume_stage, resume_sd, passes0 = (
            ck["decisions"],
            ck["stage"],
            ck["stage_decisions"],
            ck["passes"],
        )
        print(f"resumed {args.run_id} at stage {resume_stage}, {total:,} decisions", flush=True)
    elif not (run_dir / "ckpt_init.pt").exists():
        save_ckpt(
            run_dir / "ckpt_init.pt", policy, opt, stages[0], 0, 0, 0, cfg
        )  # the untrained policy, for the reel
    venv = VecLob(workers, per, cfg["seed"] + (total % 997), cur.stage_mix(stages[0]))
    metrics = open(run_dir / "metrics.jsonl", "a")
    evals = open(run_dir / "evals.jsonl", "a")
    try:
        for stage in stages:
            if resume_stage and C.STAGES.index(stage) < C.STAGES.index(resume_stage):
                continue
            oracle_rate = cur.load_oracle_rate(stage)
            gate = cur.Gate(stage, oracle_rate)
            budget = C.BUDGETS[stage]
            sd = resume_sd if stage == resume_stage else 0
            gate.passes = passes0 if stage == resume_stage else 0
            next_eval = (sd // C.GATE_EVERY + 1) * C.GATE_EVERY
            venv.set_stage_mix(cur.stage_mix(stage))
            obs = to_tensor(venv.reset())
            recent: deque = deque(maxlen=2000)
            print(
                f"== {stage}: gate {gate.threshold:.3f}, budget {budget:,} decisions, {n_envs} envs x {n_steps} steps",
                flush=True,
            )
            t_stage = time.time()
            n_updates = 0
            while True:
                frac = min(sd / budget, 1.0)
                for g in opt.param_groups:
                    g["lr"] = pp["lr"] * max(0.1, 1.0 - frac)
                t0 = time.time()
                tick0 = venv.total_ticks()
                ob = torch.zeros(n_steps, n_envs, C.OBS_DIM)
                th, lp, va, rw, dn = (torch.zeros(n_steps, n_envs) for _ in range(5))
                pr = torch.zeros(n_steps, n_envs, 3)
                for t in range(n_steps):
                    a, throw, praw, logp, val = policy.act(obs)
                    nxt, rew, term, _, infos = venv.step(a.numpy())
                    ob[t], th[t], pr[t], lp[t], va[t] = obs, throw, praw, logp, val
                    rw[t], dn[t] = to_tensor(rew), to_tensor(term)
                    for info in infos.values():
                        recent.append(info["outcome"])
                    obs = to_tensor(nxt)
                with torch.no_grad():
                    last = policy.value(obs)
                adv, ret = gae(rw, va, dn, last)

                def flat(x):
                    return x.reshape(n_steps * n_envs, *x.shape[2:])

                batch = Rollout(
                    flat(ob), flat(th), flat(pr), flat(lp), flat(adv), flat(ret), flat(va)
                )
                stats = update(
                    policy, opt, batch, pp["epochs"], pp["minibatches"], pp["clip"], pp["ent_coef"],
                    pp["vf_coef"], pp["max_grad_norm"], pp["target_kl"],
                )  # fmt: skip
                steps = n_steps * n_envs
                total += steps
                sd += steps
                n_updates += 1
                c = Counter(recent)
                n_rec = max(len(recent), 1)
                row = {
                    "stage": stage, "decisions": total, "stage_decisions": sd, "episodes_recent": len(recent),
                    "dunk": c["DUNK"] / n_rec, "outcomes": {k: v / n_rec for k, v in c.items()},
                    "reward": float(rw.sum() / max(dn.sum().item(), 1.0)), "p_throw": float(th.mean()),
                    "ticks_per_s": (venv.total_ticks() - tick0) / (time.time() - t0), **stats,
                }  # fmt: skip
                log(metrics, row)
                if n_updates % 4 == 0:
                    print(
                        f"{stage} {sd:>9,} dunk {row['dunk']:.3f} rew {row['reward']:+.3f} "
                        f"p_throw {row['p_throw']:.3f} kl {stats['kl']:.4f} {row['ticks_per_s']:,.0f} ticks/s",
                        flush=True,
                    )
                if sd >= next_eval:
                    next_eval += C.GATE_EVERY
                    ev = cur.evaluate(policy, venv, stage)
                    obs = to_tensor(venv.reset())  # evaluation leaves the workers' envs finished: start fresh episodes
                    ev.update(decisions=total, stage_decisions=sd, threshold=gate.threshold,
                              wall_min=(time.time() - t_stage) / 60)  # fmt: skip
                    log(evals, ev)
                    passed = gate.update(ev["success"])
                    shares = {k: round(v, 3) for k, v in ev["outcomes"].items()}
                    print(
                        f"  eval {stage}: DUNK {ev['success']:.3f} (gate {gate.threshold:.3f}, passes {gate.passes}) {shares}",
                        flush=True,
                    )
                    save_ckpt(
                        run_dir / f"ckpt_{stage}_{sd // 1000}k.pt",
                        policy,
                        opt,
                        stage,
                        total,
                        sd,
                        gate.passes,
                        cfg,
                    )
                    save_ckpt(latest, policy, opt, stage, total, sd, gate.passes, cfg)
                    if passed:
                        save_ckpt(
                            run_dir / f"ckpt_{stage}_final.pt",
                            policy,
                            opt,
                            stage,
                            total,
                            sd,
                            gate.passes,
                            cfg,
                        )
                        print(
                            f"== {stage} passed after {sd:,} decisions ({(time.time() - t_stage) / 60:.1f} min)",
                            flush=True,
                        )
                        break
                if sd >= budget:
                    ev = cur.evaluate(policy, venv, stage)
                    diag = {"stage": stage, "eval": ev, "recent": dict(c)}
                    (run_dir / "diagnostics.json").write_text(json.dumps(diag, indent=1))
                    print(
                        f"!! {stage} gate NOT reached within {budget:,} decisions; see diagnostics.json",
                        flush=True,
                    )
                    return 1
        return 0
    finally:
        venv.close()
        metrics.close()
        evals.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="S0,S1")
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--config", default="demos/lobpass/configs/default.yaml")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--per-worker", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    raise SystemExit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
