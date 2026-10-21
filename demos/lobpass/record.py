"""Record a play as data for the animation baker: per-tick states of everyone, the policy's decisions and the
events with their ticks (B1).

    pixi run -e lob lob-record --stage S3c --seed 12 --checkpoint runs/lobpass/main/ckpt_S3c_final.pt --out runs/lobpass/footage/ep12
    pixi run lob-record --stage S1 --seed 3 --checkpoint oracle --out runs/lobpass/footage/oracle3
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

import numpy as np

try:  # torch must load before MuJoCo's DLLs on Windows (or its own DLL load fails)
    import torch  # noqa: F401
except (
    ImportError
):  # the default pixi environment has no torch: only scripted/oracle plays work there
    pass

from . import constants as C  # noqa: E402
from .env import LobPassEnv  # noqa: E402
from .machines import DEFENDER_STATES  # noqa: E402
from .scenario import ROLE_NAMES, Scenario, sample_scenario  # noqa: E402
from .world import PLAYERS  # noqa: E402

FORMAT, VERSION = "lobpass-episode", 1
NO_THROW = np.array([-1.0, 0.0, 0.0, 0.0])


class _Grab:
    """Per-tick recorder installed as ``env.recorder``."""

    def __init__(self):
        self.rows = []

    def __call__(self, env):
        st = env.sim.state()
        sl = env.slots
        idx = [sl["ball"]] + [sl[p] for p in PLAYERS]
        beh = env.sim._beh.by_name
        self.rows.append(
            (
                env.sim.tick,
                st.body_pos[idx].copy(),
                st.body_linvel[idx].copy(),
                int(env.sim["dunker_state"]),
                [int(beh[f"d{i}"].state_idx) for i in range(C.N_DEF_SLOTS)],
            )
        )


def _tick_of(t: float) -> int:
    return int(round(float(t) / C.TICK))


def _events(env: LobPassEnv, g: _Grab, info: dict) -> list[dict]:
    s = env.sim
    ev: list[dict] = []

    def add(kind, tick, actor=None, **data):
        ev.append({"tick": int(tick), "type": kind, "actor": actor, "data": data})

    if info.get("t_release") is not None:
        rel = _tick_of(info["t_release"])
        add("windup", rel - C.WINDUP_TICKS, "passer")
        add("release", rel, "passer", v0=[float(x) for x in info["v0"]])
    for name, key in (
        ("catch", "catch"),
        ("rim_touch", "rim"),
        ("board_touch", "board"),
        ("floor_touch", "floor"),
    ):
        if s[f"ev_{key}"]:
            add(name, _tick_of(s[f"t_{key}"]), "dunker" if key == "catch" else None)
    for i in range(C.N_DEF_SLOTS):
        if s[f"ev_int{i}"]:
            add("intercept", _tick_of(s[f"t_int{i}"]), f"d{i}")
    if s["ev_blocked"]:
        add("block", _tick_of(s["t_blocked"]), "dunker")
    dstate = np.array([r[3] for r in g.rows])
    ticks = np.array([r[0] for r in g.rows])
    for state, kind in ((3, "takeoff"), (5, "land")):
        hit = np.nonzero(dstate == state)[0]
        if len(hit):
            add(kind, ticks[hit[0]], "dunker")
    for i in range(C.N_DEF_SLOTS):
        st_i = np.array([r[4][i] for r in g.rows])
        for state, kind in (
            (DEFENDER_STATES.index("jump"), "takeoff"),
            (DEFENDER_STATES.index("landed"), "land"),
        ):
            hit = np.nonzero(st_i == state)[0]
            if len(hit) and any(d.slot == i for d in env.scn.defenders):
                add(kind, ticks[hit[0]], f"d{i}")
    return sorted(ev, key=lambda e: (e["tick"], e["type"]))


def record_episode(
    scn: Scenario,
    checkpoint: str = "oracle",
    post_seconds: float = 2.5,
    env: LobPassEnv | None = None,
    script: dict | None = None,
    sample: bool = False,
    seed: int = 0,
):
    """Play ``scn`` once. ``checkpoint`` is a trained policy file, or "oracle" for the scripted expert. Returns
    (arrays dict, meta dict)."""
    env = env or LobPassEnv({scn.stage: 1.0})
    g = _Grab()
    decisions: list[dict] = []
    policy = None
    if (
        script is not None
    ):  # a hand-made play: {"release": tick, "action"|"v0": ...} or {"when": "blocked", "v0": ...}
        checkpoint = "scripted"
    elif checkpoint == "oracle":
        from . import oracle

        sol = oracle.solve(env, scn)
        script = sol if sol["success_any"] else None
    else:
        import torch

        from .train import load_policy

        policy = load_policy(checkpoint)
        torch.manual_seed(seed)
    obs, _ = env.reset(options={"scenario": scn})
    env.recorder = g
    g(env)  # tick 0
    info: dict = {}
    done = False
    while not done:
        tick = env.sim.tick
        if policy is not None:
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32)[None]
                act_t, _, _, _, val = policy.act(t_obs, deterministic=not sample)
                p_throw = float(policy.p_throw(t_obs)[0])
            action = act_t[0].numpy()
            decisions.append(
                {
                    "tick": tick,
                    "p_throw": p_throw,
                    "a": [float(x) for x in action],
                    "value": float(val[0]),
                }
            )
        else:
            go = False
            if script is not None:
                if script.get("when") == "blocked":
                    go = int(env.sim["dunker_state"]) == 6
                else:
                    go = tick == script["release"] - C.WINDUP_TICKS
            action = script["action"] if (go and "action" in script) else NO_THROW
            decisions.append(
                {
                    "tick": tick,
                    "p_throw": 1.0 if go else 0.0,
                    "a": [float(x) for x in action],
                    "value": 0.0,
                }
            )
            if go and "v0" in script:
                obs, reward, done, _, info = env.throw(np.asarray(script["v0"], float))
                continue
        obs, reward, done, _, info = env.step(action)
    for _ in range(int(post_seconds / C.TICK)):  # let the ball finish its flight
        env._tick()
    env.recorder = None
    rows = g.rows
    arrays = {
        "tick": np.array([r[0] for r in rows]),
        "pos": np.array([r[1] for r in rows]),  # (T, 1 + 7, 3): ball, then PLAYERS
        "vel": np.array([r[2] for r in rows]),
        "dunker_state": np.array([r[3] for r in rows]),
        "def_state": np.array([r[4] for r in rows]),
    }
    actors = [{"id": "passer", "team": "offense", "role": "passer", "H": scn.passer_H, "L": 0.0,
               "R_s": C.reach(scn.passer_H), "slot": 0},
              {"id": "dunker", "team": "offense", "role": "dunker", "H": scn.dunker_H, "L": scn.dunker_L,
               "R_s": scn.dunker_R_s, "slot": 1}]  # fmt: skip
    for d in scn.defenders:
        actors.append({"id": f"d{d.slot}", "team": "defense", "role": ROLE_NAMES[d.role], "H": d.H, "L": d.L,
                       "R_s": d.R_s, "slot": 2 + d.slot})  # fmt: skip
    meta = {
        "format": FORMAT, "version": VERSION, "seed": scn.seed, "stage": scn.stage, "checkpoint": str(checkpoint),
        "scenario": asdict(scn), "actors": actors,
        "plan": {"start": list(scn.dunker_start), "corner": None if scn.corner is None else list(scn.corner),
                 "takeoff": list(scn.takeoff)},
        "decisions": decisions, "events": _events(env, g, info), "outcome": info.get("outcome"),
        "reward": float(reward), "hold": [float(x) for x in env.hold], "players": list(PLAYERS), "dt": C.TICK,
    }  # fmt: skip
    return arrays, meta


def save_episode(arrays: dict, meta: dict, out: str | pathlib.Path) -> pathlib.Path:
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "episode.npz", **arrays)
    (out / "episode.json").write_text(json.dumps(meta, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="S1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", default="oracle")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    arrays, meta = record_episode(sample_scenario(args.stage, args.seed), args.checkpoint)
    path = save_episode(arrays, meta, args.out)
    print(f"{path}: {meta['outcome']} ({len(arrays['tick'])} ticks, {len(meta['events'])} events)")


if __name__ == "__main__":
    main()
