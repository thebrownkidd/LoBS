"""Scripted expert: finds a release that dunks, for the feasibility checks and the training gates (A10).

1. record the no-throw play, 2. enumerate catch targets in both catch windows and release ticks (decision tick +
15, decisions every 10 ticks), 3. solve release velocities with the batch drag solver, 4. rank by clearance from the
defenders' reach cylinders, 5. confirm the best candidates closed-loop in LoBS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import ballistics as B
from . import constants as C
from .env import LobPassEnv
from .scenario import Scenario

FLIGHTS = tuple(range(50, 171, 10))  # ticks from the start of the impulse tick to the catch
WINDOW_STEP = 3


@dataclass
class Rec:
    dk_pos: np.ndarray  # (T,3)
    dk_vel: np.ndarray
    dk_state: np.ndarray  # (T,)
    def_pos: np.ndarray  # (T,5,3)


def record(env: LobPassEnv, scn: Scenario, max_ticks: int = 800) -> Rec:
    """The no-throw play, one entry per tick (index = tick)."""
    env._start(scn)
    sim, slots = env.sim, env.slots
    names = [f"d{i}" for i in range(C.N_DEF_SLOTS)]

    def grab():
        st = sim.state()
        return (st.body_pos[slots["dunker"]].copy(), st.body_linvel[slots["dunker"]].copy(),
                int(sim["dunker_state"]), np.array([st.body_pos[slots[n]] for n in names]))  # fmt: skip

    rows, down = [grab()], 0
    for _ in range(max_ticks):
        sim.step()
        rows.append(grab())
        down = down + 1 if rows[-1][2] in (5, 6) else 0
        if down > 30:
            break
    return Rec(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
               np.array([r[2] for r in rows]), np.array([r[3] for r in rows]))  # fmt: skip


def to_action(env: LobPassEnv, v0) -> np.ndarray:
    """Inverse of ``env.decode``: the action producing release velocity v0 (entries outside [-1,1] mean it is not
    representable). The throw component is +1."""
    v0 = np.asarray(v0, float)
    d = np.array(env.scn.takeoff) - env.hold[:2]
    psi_ref = math.atan2(d[1], d[0])
    dpsi = (math.atan2(v0[1], v0[0]) - psi_ref + math.pi) % (2 * math.pi) - math.pi
    h_ap = env.hold[2] + v0[2] ** 2 / (2 * C.G)
    t_f = v0[2] / C.G + math.sqrt(2.0 * max(h_ap - C.RIM_TOP_Z, 1e-6) / C.G)
    rng_xy = float(np.hypot(v0[0], v0[1])) * t_f
    return np.array([1.0, dpsi / C.PSI_SPAN, (h_ap - C.APEX_MIN) / C.APEX_SPAN - 1.0,
                     (rng_xy / float(np.linalg.norm(d)) - 1.0) / C.RANGE_SPAN])  # fmt: skip


def representable(a: np.ndarray) -> bool:
    return bool(np.all(np.abs(a[1:]) <= 1.0))


def candidates(env: LobPassEnv, scn: Scenario, rec: Rec):
    """All feasible releases: arrays v0 (N,3), release_tick (N,), catch_tick (N,), kind (N,), clearance (N,)."""
    hold = env.hold
    T = np.array(scn.takeoff)
    n_t = len(rec.dk_state)
    targets, nflight, mticks, kinds = [], [], [], []
    for m in range(20, n_t, WINDOW_STEP):
        st, p, v = rec.dk_state[m], rec.dk_pos[m], rec.dk_vel[m]
        sp = float(np.hypot(v[0], v[1]))
        fwd = v[:2] / sp if sp > 1e-6 else np.zeros(2)
        axis = p[:2] + C.CATCH_AXIS_FWD * fwd
        if st in (1, 2) and float(np.linalg.norm(p[:2] - T)) <= C.DUNKABLE_GROUND_T_DIST:
            z, kind = p[2] + 1.75, 0
        elif st in (3, 4) and p[2] >= C.DUNKABLE_AIR_JZ:
            z = max(p[2] + C.CATCH_Z[1] * scn.dunker_H - 0.25, C.DUNKABLE_AIR_BALL_Z + 0.1)
            if z > p[2] + C.CATCH_Z[1] * scn.dunker_H - 0.08:
                continue
            kind = 1
        else:
            continue
        for n in FLIGHTS:
            r = m - n  # impulse tick start; decision tick = r - 15 must sit on the 10-tick grid
            if r >= C.WINDUP_TICKS and (r - C.WINDUP_TICKS) % C.DECISION_TICKS == 0:
                targets.append([axis[0], axis[1], z])
                nflight.append(n)
                mticks.append(m)
                kinds.append(kind)
    if not targets:
        return None
    targets, nflight, mticks = np.array(targets), np.array(nflight), np.array(mticks)
    v0, miss = B.solve_release_batch(hold, targets, nflight)
    ok = (miss < 0.02) & (np.linalg.norm(v0, axis=1) <= C.MAX_THROW_SPEED)
    if not ok.any():
        return None
    targets, nflight, mticks, v0 = targets[ok], nflight[ok], mticks[ok], v0[ok]
    kinds = np.array(kinds)[ok]
    release = mticks - nflight
    samples = (nflight[:, None] * np.array([0.25, 0.5, 0.75, 0.9])).astype(int)
    samples = np.maximum(samples, 1)
    path = B.batch_positions(hold, v0, nflight, samples)[:, 1:]  # (N, S, 3) at the sampled ticks
    clr = np.full(len(v0), 9.0)
    for i in range(C.N_DEF_SLOTS):
        d = next((x for x in scn.defenders if x.slot == i), None)
        if d is None:
            continue
        for k in range(samples.shape[1]):
            tick = np.minimum(release + samples[:, k], len(rec.def_pos) - 1)
            dp = rec.def_pos[tick, i]  # (N,3)
            horiz = np.linalg.norm(path[:, k, :2] - dp[:, :2], axis=1) - C.INT_RADIUS
            vert = path[:, k, 2] - (dp[:, 2] + d.R_s) - 0.12
            clr = np.minimum(clr, np.maximum(horiz, vert))
    return {
        "v0": v0,
        "release": release,
        "catch": mticks,
        "kind": kinds,
        "clearance": clr,
        "n": nflight,
    }


def _closed_loop(env: LobPassEnv, scn: Scenario, v0, release_tick: int) -> str:
    env.reset(options={"scenario": scn})
    for _ in range((release_tick - C.WINDUP_TICKS) // C.DECISION_TICKS):
        env.step(np.array([-1.0, 0.0, 0.0, 0.0]))
    return env.throw(v0)[4]["outcome"]


def solve(env: LobPassEnv, scn: Scenario, top_k: int = 16) -> dict:
    """Best dunking release for ``scn`` (or the failure). ``representable`` tells whether the agent's action box
    contains it; the representable set is tried first."""
    rec = record(env, scn)
    cand = candidates(env, scn, rec)
    out = {
        "success": False,
        "n_candidates": 0,
        "success_any": False,
        "representable_success": False,
    }
    if cand is None:
        return out
    out["n_candidates"] = len(cand["v0"])
    acts = np.array([to_action(env, v) for v in cand["v0"]])
    rep = np.array([representable(a) for a in acts])
    # rank: clearance (capped), then shorter flights, then slower throws
    score = (
        -np.minimum(cand["clearance"], 1.0)
        + 0.002 * cand["n"]
        + 0.01 * np.linalg.norm(cand["v0"], axis=1)
    )
    for label, mask in (("rep", rep), ("any", np.ones_like(rep))):
        idx = [i for i in np.argsort(score) if mask[i]][:top_k]
        for i in idx:
            if _closed_loop(env, scn, cand["v0"][i], int(cand["release"][i])) == "DUNK":
                out.update(success_any=True, v0=cand["v0"][i], release=int(cand["release"][i]), action=acts[i],
                           kind=int(cand["kind"][i]), representable_success=(label == "rep"))  # fmt: skip
                out["success"] = out["representable_success"]
                return out
    return out


# ---------------------------------------------------------------------- table over the fixed eval set
_ENV: LobPassEnv | None = None


def _solve_one(args):
    global _ENV
    stage, seed = args
    if _ENV is None:
        _ENV = LobPassEnv({stage: 1.0})
    from .scenario import sample_scenario

    r = solve(_ENV, sample_scenario(stage, seed))
    return {"seed": seed, "success": r["success"], "success_any": r["success_any"], "n": r["n_candidates"],
            "kind": r.get("kind", -1)}  # fmt: skip


def eval_seeds(stage: str, n: int = C.EVAL_N) -> list[int]:
    """The fixed evaluation scenarios of a stage (A10)."""
    return [C.EVAL_SEED + 10_000 * C.STAGES.index(stage) + i for i in range(n)]


def table(stage: str, n: int = C.EVAL_N, workers: int = 12) -> dict:
    """Oracle success over the stage's eval set (cached by the CLI under runs/lobpass/oracle_<stage>.json)."""
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(stage, s) for s in eval_seeds(stage, n)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_solve_one, jobs, chunksize=4))
    ok = np.array([r["success"] for r in rows])
    any_ = np.array([r["success_any"] for r in rows])
    return {
        "stage": stage, "n": n, "oracle": float(ok.mean()), "oracle_any": float(any_.mean()),
        "representable_share": float(ok.sum() / max(any_.sum(), 1)),
        "air_share": float(np.mean([r["kind"] == 1 for r in rows if r["success"]] or [0.0])),
        "failures": [r["seed"] for r in rows if not r["success"]][:50],
    }  # fmt: skip


def main() -> None:
    import argparse
    import json
    import pathlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["S0"])
    ap.add_argument("--n", type=int, default=C.EVAL_N)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    out = pathlib.Path("runs/lobpass")
    out.mkdir(parents=True, exist_ok=True)
    for st in args.stage:
        res = table(st, args.n, args.workers)
        (out / f"oracle_{st}.json").write_text(json.dumps(res, indent=1))
        print(f"{st}: oracle {res['oracle']:.3f} (any {res['oracle_any']:.3f}, representable share "
              f"{res['representable_share']:.3f}, air share {res['air_share']:.2f})", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
