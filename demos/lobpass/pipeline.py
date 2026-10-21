"""The whole curriculum, resumable: for every stage clone the oracle on the new scenarios (mixed with a little of the
earlier stages), then fine-tune with PPO against the real environment until the stage's gate passes twice.

    pixi run -e lob python -m demos.lobpass.pipeline                # every stage S0 .. S3c
    pixi run -e lob python -m demos.lobpass.pipeline --from S3a     # restart from a stage

Run directories: runs/lobpass/ft_<stage>/ (checkpoints, metrics.jsonl, evals.jsonl). A stage whose
``ckpt_<stage>_final.pt`` exists is skipped."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

from . import constants as C

ROOT = pathlib.Path("runs/lobpass")
# stage -> BC spec (scenarios to try; some stages have unsolvable ones) and the earlier stages mixed in
BC_SPECS = {
    "S2": ["S2:10000", "S1:800", "S0:600"],
    "S3a": ["S3a:10000", "S2:1500", "S1:600"],
    "S3b": ["S3b:18000", "S3a:1500", "S2:800"],
    "S3c": ["S3c:30000", "S3b:5000", "S3a:1000"],
}
BASE = (
    ROOT / "bcmain" / "ckpt_bc.pt"
)  # S0 + S1 cloned from the oracle (95.9 % / 94.9 % before any PPO)


def run(cmd: list[str], log: pathlib.Path) -> int:
    print("$", " ".join(cmd), flush=True)
    with open(log, "a", encoding="utf-8") as fp:
        fp.write("$ " + " ".join(cmd) + "\n")
        fp.flush()
        return subprocess.call(cmd, stdout=fp, stderr=subprocess.STDOUT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="S0")
    ap.add_argument("--to", dest="stop", default="S3c")
    args = ap.parse_args()
    log = ROOT / "pipeline.log"
    py = [sys.executable, "-u", "-m"]
    stages = C.STAGES[C.STAGES.index(args.start) : C.STAGES.index(args.stop) + 1]
    prev_final = None
    for st in stages:
        run_id = "ft_S0S1" if st in ("S0", "S1") else f"ft_{st}"
        final = ROOT / run_id / f"ckpt_{('S1' if st in ('S0', 'S1') else st)}_final.pt"
        if st == "S0":
            init = str(BASE)
            stage_arg = "S0,S1"
        elif st == "S1":
            prev_final = final
            continue  # trained together with S0
        else:
            prev = C.STAGES[C.STAGES.index(st) - 1]
            prev_run = "ft_S0S1" if prev in ("S0", "S1") else f"ft_{prev}"
            src = ROOT / prev_run / f"ckpt_{'S1' if prev in ('S0', 'S1') else prev}_final.pt"
            bc_out = ROOT / run_id / "ckpt_bc.pt"
            if not final.exists() and not bc_out.exists():
                (ROOT / run_id).mkdir(parents=True, exist_ok=True)
                rc = run(py + ["demos.lobpass.bc", "--spec", *BC_SPECS[st], "--run-id", run_id, "--init", str(src),
                               "--epochs", "80", "--lr", "5e-4", "--pos-weight", "3", "--workers", "14", "--eval-stages", st], log)  # fmt: skip
                if rc:
                    return rc
            init, stage_arg = str(bc_out), st
        if final.exists():
            print(f"{st}: already done ({final})", flush=True)
            continue
        rc = run(py + ["demos.lobpass.train", "--stages", stage_arg, "--run-id", run_id, "--init", init,
                       "--lr", "2e-5", "--ent", "0", "--warmup", "6", "--max-stage-decisions", "500000", "--accept-best", "--resume"], log)  # fmt: skip
        if rc or not final.exists():
            print(f"!! {st}: gate not reached (rc={rc}); see {ROOT / run_id}", flush=True)
            return rc or 1
        prev_final = final
    _ = prev_final
    print("pipeline complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
