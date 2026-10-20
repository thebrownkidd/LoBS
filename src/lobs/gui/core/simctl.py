"""SimController: owns the live Simulation for the GUI (no Qt; a QTimer calls ``advance``)."""

from __future__ import annotations

import time

from ... import backends


class SimController:
    def __init__(self, session):
        self.session = session
        self.backend = "mujoco"
        self.seed = 0
        self.speed = 1.0  # sim seconds per wall second; 0 = as fast as the budget allows
        self.playing = False
        self.sim = None
        self.error: str | None = None
        self.realtime_factor = 0.0
        self._listeners: list = []
        self._wall_ref = None
        self._sim_ref = 0.0
        session.subscribe(self._on_session)

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def _emit(self, kind: str) -> None:
        for fn in list(self._listeners):
            fn(kind)

    def available_backends(self) -> list[str]:
        return list(backends.available())

    # ------------------------------------------------------------ lifecycle
    def _on_session(self, kind: str) -> None:
        if kind in ("load", "undo", "redo", "edit"):
            self.playing = False
            self.close()
            self._emit("invalidated")

    def build(self) -> bool:
        """(Re)create the simulation from the session world. Returns False and sets ``error`` on failure."""
        self.close()
        try:
            self.sim = self.session.world.simulate(self.backend, seed=self.seed, record=True)
            self.error = None
        except Exception as e:  # noqa: BLE001 - shown in the status bar / validation panel
            self.sim, self.error = None, str(e)
        self._wall_ref = None
        self._emit("built")
        return self.sim is not None

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def ensure(self) -> bool:
        return self.sim is not None or self.build()

    def reset(self) -> None:
        if not self.ensure():
            return
        self.sim.reset(seed=self.seed)
        self._wall_ref = None
        self._emit("reset")

    def set_backend(self, name: str) -> None:
        self.backend = name
        self.playing = False
        self.build()

    # ------------------------------------------------------------ running
    def play(self) -> None:
        if self.ensure():
            self.playing = True
            self._wall_ref = None
            self._emit("play")

    def pause(self) -> None:
        self.playing = False
        self._emit("pause")

    def step(self, n: int = 1) -> int:
        if not self.ensure():
            return 0
        self.sim.step(n=n)
        self._emit("stepped")
        return n

    def advance(self, wall_budget: float = 0.012) -> int:
        """Advance toward real time x speed, spending at most ``wall_budget`` seconds. Returns ticks run."""
        if not (self.playing and self.sim is not None):
            return 0
        sim = self.sim
        now = time.perf_counter()
        if self._wall_ref is None:
            self._wall_ref, self._sim_ref = now, sim.time
        if self.speed <= 0:
            want = sim.tick_dt * 4
        else:
            want = (now - self._wall_ref) * self.speed + self._sim_ref - sim.time
        ticks, start = 0, time.perf_counter()
        while want > 0 and time.perf_counter() - start < wall_budget:
            sim.step()
            ticks += 1
            want -= sim.tick_dt
            if sim.done:
                self.playing = False
                break
        spent = time.perf_counter() - start
        if ticks and spent > 0:
            self.realtime_factor = 0.8 * self.realtime_factor + 0.2 * (ticks * sim.tick_dt / spent)
        if want > 0.25 and self.speed > 0:  # cannot keep up: resync instead of sprinting forever
            self._wall_ref, self._sim_ref = time.perf_counter(), sim.time
        if ticks:
            self._emit("stepped")
        return ticks
