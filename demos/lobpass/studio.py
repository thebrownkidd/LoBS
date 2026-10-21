"""Run the trained passer inside LoBS Studio.

The passer's brain is a LoBS script behavior (``lobpass.passer``): every decision tick it feeds the policy the same
109 observations the trainer used, and when it decides to throw it winds up and applies the release force on the ball.
Everything else (the dunker and defenders) is the declarative state machines of the training world, so you can watch
and edit them live in the behavior editor.

    pixi run -e lob python -m demos.lobpass.studio --checkpoint runs/lobpass/ft_S3c/ckpt_S3c_final.pt --stage S3c --seed 3
    pixi run -e lob python -m demos.lobpass.studio --checkpoint ... --check        # headless: run the play and print it
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
import types

import numpy as np

try:  # torch must load before MuJoCo's DLLs on Windows (or its own DLL load fails)
    import torch  # noqa: F401
except (
    ImportError
):  # the default pixi environment has no torch: only scripted/oracle plays work there
    pass

import lobs  # noqa: E402

from . import constants as C  # noqa: E402
from . import obs as O  # noqa: E402
from .env import decode_action  # noqa: E402
from .scenario import Scenario, sample_scenario  # noqa: E402
from .world import PLAYERS, build_world, to_reset_args  # noqa: E402

NAMES = ["ball", *PLAYERS]
_POLICIES: dict = {}


def _policy(path: str):
    if path not in _POLICIES:
        from .train import load_policy

        _POLICIES[path] = load_policy(path)
    return _POLICIES[path]


@lobs.behavior("lobpass.passer")
class PasserBrain:
    """Script behavior: decisions every 10 ticks, wind-up of 15 ticks, one release force (A9)."""

    def reset(self, ctx):
        import torch

        self.torch = torch
        self.policy = _policy(ctx.params["checkpoint"])
        self.scn: Scenario = sample_scenario(ctx.params["stage"], int(ctx.params["seed"]))
        init, _ = to_reset_args(self.scn)
        self.hold = np.array(init["ball"].pos, dtype=float)
        self.v0 = None
        self.release_tick = None
        self.p_throw = 0.0

    def _observe(self, ctx) -> np.ndarray:
        pos, vel = np.zeros((len(NAMES), 3)), np.zeros((len(NAMES), 3))
        for i, n in enumerate(NAMES):
            thing = "ball" if n == "ball" else f"{n}/body:origin"
            pos[i], vel[i] = ctx.pos(thing), ctx.vel(thing)
        frame = types.SimpleNamespace(body_pos=pos, body_linvel=vel)
        slots = {n: i for i, n in enumerate(NAMES)}
        return O.build(frame, slots, self.scn, int(ctx.measure("dunker_state")), self.hold, ctx.t)

    def step(self, ctx):
        if self.v0 is None:
            if ctx.tick % C.DECISION_TICKS == 0:
                with self.torch.no_grad():
                    x = self.torch.as_tensor(self._observe(ctx), dtype=self.torch.float32)[None]
                    action, *_ = self.policy.act(x, deterministic=True)
                    self.p_throw = float(self.policy.p_throw(x)[0])
                if action[0, 0] >= 0.0:
                    self.v0, _ = decode_action(self.scn, self.hold, action[0].numpy().astype(float))
                    self.release_tick = ctx.tick + C.WINDUP_TICKS
        elif ctx.tick == self.release_tick:
            ctx.force("ball", C.BALL_MASS * (self.v0 - ctx.vel("ball")) / C.TICK)


def build_show_world(checkpoint: str, stage: str, seed: int) -> lobs.World:
    """The training world, set to scenario (stage, seed), with the trained passer attached."""
    w = build_world("train")
    scn = sample_scenario(stage, seed)
    init, rnd = to_reset_args(scn)
    for name, ini in init.items():
        actor = w._actor(name)
        if ini.pos is not None:
            actor.initial.pose = lobs.schema.Pose(pos=tuple(ini.pos), quat=actor.pose.quat)
        if ini.linvel is not None:
            actor.initial.linvel = tuple(ini.linvel)
        actor.initial.q.update(ini.q or {})
    for b in w.spec.behaviors:
        for k, v in rnd["behaviors"].get(b.name, {}).get("variables", {}).items():
            b.machine.variables[k] = v if np.ndim(v) == 0 else tuple(float(x) for x in v)
    w.add_behavior("brain", on="passer", script="lobpass.passer", period=C.TICK,
                   params={"checkpoint": str(checkpoint), "stage": stage, "seed": int(seed)})  # fmt: skip
    return w


def check(checkpoint: str, stage: str, seed: int, seconds: float = 7.0) -> dict:
    """Headless run of the studio world: does the passer throw, and what happens?"""
    w = build_show_world(checkpoint, stage, seed)
    sim = w.simulate("mujoco", record=False)
    sim.run(seconds=seconds)
    return {"released": bool(sim["released"]), "catch": bool(sim["ev_catch"]), "intercept": any(bool(sim[f"ev_int{i}"]) for i in range(C.N_DEF_SLOTS)),
            "t": sim.time, "native_core": sim.info["native_core"]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--stage", default="S3c")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument(
        "--check",
        action="store_true",
        help="run headless and print the result instead of opening the window",
    )
    args = ap.parse_args()
    if args.check:
        print(check(args.checkpoint, args.stage, args.seed))
        return
    from PySide6.QtWidgets import QApplication

    from lobs.gui.app import apply_theme, make_window

    path = pathlib.Path(tempfile.mkdtemp()) / "lobpass_show.yaml"
    build_show_world(args.checkpoint, args.stage, args.seed).save(path)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    apply_theme(app)
    win = make_window(path)
    win.resize(1500, 900)
    win.show()
    os._exit(app.exec())


if __name__ == "__main__":
    main()
