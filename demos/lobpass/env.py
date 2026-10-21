"""LobPassEnv: the passer's decision problem on top of the LoBS lob-pass world (A8, A9).

One ``step`` is a 0.1 s decision. A throw action runs the whole play (wind-up, release, flight, resolution) inside
that step and terminates the episode; the only reward is the outcome table in constants.REWARD."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import ballistics as B
from . import constants as C
from . import obs as O
from .scenario import Scenario, sample_scenario
from .world import build_world, dyn_slots, to_reset_args

EVENT_KINDS = ("int", "catch", "rim", "board", "floor", "out")  # tie order: earlier wins


class LobPassEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, stage_mix: dict | None = None, base_seed: int = 0, rank: int = 0):
        super().__init__()
        self.world = build_world("train")
        self.sim = self.world.simulate("mujoco", seed=0, record=False)
        self.slots = dyn_slots(self.sim)
        self.observation_space = spaces.Box(-5.0, 5.0, (C.OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (C.ACTION_DIM,), np.float32)
        self.stage_mix = dict(stage_mix or {"S0": 1.0})
        self.base_seed, self.rank, self.episode = base_seed, rank, 0
        self.rng = np.random.default_rng(base_seed + rank * 7919)
        self.scn: Scenario | None = None
        self.hold = np.zeros(3)
        self._ticks_done = 0
        self.recorder = None  # optional callable(env), called after every tick (used by record.py)

    # ------------------------------------------------------------------ configuration
    def set_stage_mix(self, mix: dict) -> None:
        self.stage_mix = dict(mix)

    def _pick_stage(self) -> str:
        names = list(self.stage_mix)
        p = np.array([self.stage_mix[n] for n in names], dtype=float)
        return names[int(self.rng.choice(len(names), p=p / p.sum()))]

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        if seed is not None:  # an explicit seed pins the whole episode (stage pick and scenario)
            self.rng = np.random.default_rng(seed)
        if "scenario" in opts:
            scn = opts["scenario"]
        else:
            stage = opts.get("stage") or self._pick_stage()
            sseed = opts.get("scenario_seed")
            if sseed is None:
                sseed = (
                    seed if seed is not None else self.base_seed + self.rank * 10**9 + self.episode
                )
            scn = sample_scenario(stage, int(sseed))
        self.episode += 1
        self._start(scn)
        return self._obs(), {"stage": scn.stage, "scenario_seed": scn.seed}

    def _start(self, scn: Scenario) -> None:
        self._ticks_done += self.sim.tick
        self.scn = scn
        init, rnd = to_reset_args(scn)
        self.sim.reset(seed=scn.seed, initial_state=init, randomize=rnd)
        self.hold = np.array(init["ball"].pos, dtype=float)
        self.t_release: float | None = None
        self.v_release = None

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if a[0] >= 0.0:
            v0, clamped = self.decode(a)
            return self.throw(v0, clamped=clamped)
        for _ in range(C.DECISION_TICKS):
            self._tick()
        if self._no_pass():
            return self._finish("NO_PASS")
        return self._obs(), 0.0, False, False, {}

    # ------------------------------------------------------------------ actions
    def decode(self, a: np.ndarray) -> tuple[np.ndarray, bool]:
        """Action -> release velocity (drag-free inverse, speed-capped)."""
        T = np.array(self.scn.takeoff)
        d = T - self.hold[:2]
        psi = math.atan2(d[1], d[0]) + C.PSI_SPAN * a[1]
        h_ap = C.APEX_MIN + C.APEX_SPAN * (a[2] + 1.0)
        rng_xy = float(np.linalg.norm(d)) * (1.0 + C.RANGE_SPAN * a[3])
        v0, _ = B.drag_free_v0(self.hold[2], psi, h_ap, rng_xy)
        return B.clamp_speed(v0)

    def throw(self, v0, clamped: bool = False):
        """Wind up, release with velocity ``v0`` and play the flight to its outcome (terminates the episode)."""
        for _ in range(C.WINDUP_TICKS):
            self._tick()
        vb = self._ball_vel()
        v0 = np.asarray(v0, dtype=float)
        self.sim.apply_force("ball", C.BALL_MASS * (v0 - vb) / C.TICK, ticks=1)
        self.t_release, self.v_release = self.sim.time, v0.copy()
        self._tick()
        outcome = None
        while outcome is None:
            outcome = self._resolve()
            if outcome is None and self.sim.time - self.t_release >= C.MISS_TIMEOUT:
                outcome = "BLOCKED" if self.sim["ev_blocked"] else "MISS"
            if outcome is None:
                for _ in range(5):
                    self._tick()
        return self._finish(outcome, clamped=clamped)

    # ------------------------------------------------------------------ outcome
    def _resolve(self) -> str | None:
        """Earliest latched terminal event wins (ties: interception > catch > rim > board > floor > out)."""
        s = self.sim
        cands = []
        for i in range(C.N_DEF_SLOTS):
            if s[f"ev_int{i}"]:
                cands.append((float(s[f"t_int{i}"]), 0))
        for rank, kind in enumerate(EVENT_KINDS[1:], start=1):
            if s[f"ev_{kind}"]:
                cands.append((float(s[f"t_{kind}"]), rank))
        if not cands:
            return None
        t, rank = min(cands)
        if s["ev_blocked"] and float(s["t_blocked"]) < t:
            return "BLOCKED"
        kind = EVENT_KINDS[rank]
        if kind == "int":
            return "INTERCEPTED"
        if kind == "catch":
            return "DUNK" if s["dunkable"] > 0.5 else "CATCH_NO_DUNK"
        return "MISS"

    def _no_pass(self) -> bool:
        return self.sim.time >= C.NO_PASS_TIMEOUT - 1e-9 or int(self.sim["dunker_state"]) in (5, 6)

    def _reward(self, outcome: str) -> float:
        if outcome == "MISS":
            d = min(max(float(self.sim["catch_dmin"]) / C.MISS_NORM, 0.0), 1.0)
            return C.MISS_BASE + C.MISS_SCALE * d
        return C.REWARD[outcome]

    def _finish(self, outcome: str, clamped: bool = False):
        info = {
            "outcome": outcome,
            "outcome_code": C.OUTCOMES.index(outcome),
            "stage": self.scn.stage,
            "scenario_seed": self.scn.seed,
            "t_release": self.t_release,
            "v0": None if self.v_release is None else self.v_release.copy(),
            "clamped": clamped,
            "ticks": self.sim.tick,
            "n_def": self.scn.n_def,
        }
        if outcome in ("DUNK", "CATCH_NO_DUNK"):  # caught on the run, or in the air (jump / air states)
            info["finish"] = "air" if int(self.sim["dunker_state"]) in (3, 4) else "ground"
        if outcome == "MISS":  # what the ball hit first
            hits = [(float(self.sim[f"t_{k}"]), k) for k in ("rim", "board", "floor", "out") if self.sim[f"ev_{k}"]]
            info["miss_kind"] = min(hits)[1] if hits else "timeout"
            info["miss_d"] = float(self.sim["catch_dmin"])
        return self._obs(), self._reward(outcome), True, False, info

    # ------------------------------------------------------------------ helpers
    def _tick(self) -> None:
        self.sim.step()
        if self.recorder is not None:
            self.recorder(self)

    def total_ticks(self) -> int:
        """Simulated ticks since construction (for throughput measurement)."""
        return self._ticks_done + self.sim.tick

    def _ball_vel(self) -> np.ndarray:
        return self.sim.state().body_linvel[self.slots["ball"]].copy()

    def _obs(self) -> np.ndarray:
        return O.build(
            self.sim.state(),
            self.slots,
            self.scn,
            int(self.sim["dunker_state"]),
            self.hold,
            self.sim.time,
        )

    def close(self):
        self.sim.close()


def make_env(rank: int, base_seed: int = 0, stage_mix: dict | None = None):
    """Spawn-safe factory for ``AsyncVectorEnv`` (a module-level function returning a closure)."""

    def _make():
        return LobPassEnv(stage_mix=stage_mix, base_seed=base_seed, rank=rank)

    return _make
