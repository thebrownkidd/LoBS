"""Evaluate a checkpoint on the fixed per-stage scenario sets: success, outcome shares, how it succeeds (air or on the
run), how it fails (with example seeds) and where it stands against the oracle.

    pixi run -e lob lob-eval --checkpoint runs/lobpass/ft_S3c/ckpt_S3c_final.pt --stages S0 S1 S2 S3a S3b S3c --out runs/lobpass/eval_final
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict

import numpy as np

from . import constants as C
from . import curriculum as cur


def summarize(ev: dict, stage: str) -> dict:
    infos = ev["infos"]
    n = max(len(infos), 1)
    by_def: dict = defaultdict(lambda: Counter())
    for i in infos:
        by_def[i["n_def"]]["n"] += 1
        by_def[i["n_def"]]["dunk"] += i["outcome"] == "DUNK"
    finish = Counter(i["finish"] for i in infos if i["outcome"] == "DUNK" and "finish" in i)
    miss_kind = Counter(i.get("miss_kind") for i in infos if i["outcome"] == "MISS")
    examples = defaultdict(list)
    for i in infos:
        if i["outcome"] != "DUNK" and len(examples[i["outcome"]]) < 10:
            examples[i["outcome"]].append(i["scenario_seed"])
    oracle = cur.load_oracle_rate(stage)
    return {
        "stage": stage, "n": n, "success": ev["success"], "oracle": oracle,
        "vs_oracle": None if not oracle else ev["success"] / oracle,
        "outcomes": ev["outcomes"], "reward": ev["reward"], "mean_release_s": ev["release_time"],
        "clamp_rate": ev["clamp_rate"],
        "dunk_by_defenders": {str(k): v["dunk"] / v["n"] for k, v in sorted(by_def.items())},
        "episodes_by_defenders": {str(k): v["n"] for k, v in sorted(by_def.items())},
        "finish": {k: v / max(sum(finish.values()), 1) for k, v in finish.items()},
        "miss_kind": {str(k): v / max(sum(miss_kind.values()), 1) for k, v in miss_kind.items()},
        "failure_examples": dict(examples),
        "mean_miss_distance": float(np.mean([i["miss_d"] for i in infos if i["outcome"] == "MISS" and "miss_d" in i] or [0.0])),
    }  # fmt: skip


def plots(results: list[dict], out: pathlib.Path) -> list[pathlib.Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    stages = [r["stage"] for r in results]
    order = ["DUNK", "CATCH_NO_DUNK", "MISS", "INTERCEPTED", "BLOCKED", "NO_PASS"]
    colors = {"DUNK": "#27c46b", "CATCH_NO_DUNK": "#9bd9b4", "MISS": "#8b93a1", "INTERCEPTED": "#e5484d", "BLOCKED": "#f08a4b", "NO_PASS": "#444"}
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(len(results))
    for k in order:
        vals = np.array([r["outcomes"].get(k, 0.0) for r in results])
        ax.bar(stages, vals, bottom=bottom, label=k, color=colors[k])
        bottom += vals
    ax.plot(stages, [r["oracle"] or np.nan for r in results], "k--o", label="oracle DUNK rate")
    ax.set_ylabel("share of plays"); ax.set_ylim(0, 1.05); ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.set_title("Outcomes per stage (deterministic policy, fixed evaluation sets)")
    fig.tight_layout()
    p = out / "outcomes.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> None:
    from .train import load_policy
    from .vec import VecLob

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stages", nargs="+", default=list(C.STAGES))
    ap.add_argument("--n", type=int, default=C.EVAL_N)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy(args.checkpoint)
    venv = VecLob(args.workers, 4, 0, {args.stages[0]: 1.0})
    results = []
    try:
        for st in args.stages:
            ev = cur.evaluate(policy, venv, st, n=args.n, keep_infos=True)
            r = summarize(ev, st)
            results.append(r)
            print(f"{st}: DUNK {r['success']:.3f}" + (f" ({r['vs_oracle']:.2f} of oracle {r['oracle']:.3f})" if r["oracle"] else "")
                  + f"  {({k: round(v, 3) for k, v in r['outcomes'].items()})}", flush=True)  # fmt: skip
    finally:
        venv.close()
    (out / "metrics.json").write_text(json.dumps({"checkpoint": args.checkpoint, "stages": results}, indent=1))
    plots(results, out)
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
