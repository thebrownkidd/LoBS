"""Power tools behind the GUI panels (no Qt): world check, engine comparison, Monte Carlo, console, autosave."""

from __future__ import annotations

import contextlib
import io
import traceback
from pathlib import Path

import numpy as np

from ... import act, ops, props, rel
from ...errors import LobsError


# ------------------------------------------------------------------ check / compare
def run_check(world, backend: str = "mujoco", seconds: float = 2.0):
    from ...check import check_world

    return check_world(world, backend, seconds=seconds)


def run_compare(world, backends=("mujoco", "chrono"), seconds: float = 2.0):
    from ...compare import compare

    return compare(world, list(backends), seconds=seconds)


# ------------------------------------------------------------------ Monte Carlo
STATS = {
    "final": lambda a: a[-1], "max": np.max, "min": np.min, "mean": np.mean, "any": np.any,
}  # fmt: skip


def monte_carlo(world, measure: str, *, runs: int = 20, seconds: float = 2.0, backend: str = "mujoco",
                randomize: dict | None = None, stat: str = "final", base_seed: int = 0, progress=None) -> dict:  # fmt: skip
    """Run ``runs`` seeded episodes and summarize one measure per episode.

    Returns ``{'values', 'seeds', 'stat', 'measure', 'histogram': (counts, edges)}``. Each episode is
    ``sim.reset(seed=..., randomize=...)`` so any bar of the histogram can be reproduced from its seed."""
    if stat not in STATS:
        raise LobsError(f"stat must be one of {sorted(STATS)}")
    sim = world.simulate(backend, seed=base_seed, record=True)
    values, seeds = [], []
    try:
        for k in range(runs):
            seed = base_seed + k
            sim.reset(seed=seed, randomize=randomize)
            sim.run(seconds=seconds)
            series = np.asarray(sim.history[measure], dtype=float)
            values.append(float(STATS[stat](series)))
            seeds.append(seed)
            if progress is not None:
                progress(k + 1, runs)
    finally:
        sim.close()
    vals = np.asarray(values)
    counts, edges = np.histogram(vals, bins=min(max(runs // 2, 3), 20))
    return {
        "values": vals,
        "seeds": seeds,
        "stat": stat,
        "measure": measure,
        "histogram": (counts, edges),
    }


def seeds_in_bin(result: dict, i: int) -> list[int]:
    """Seeds whose value falls in histogram bin ``i`` (to replay that outcome)."""
    edges = result["histogram"][1]
    v = result["values"]
    last = i == len(edges) - 2
    mask = (v >= edges[i]) & ((v <= edges[i + 1]) if last else (v < edges[i + 1]))
    return [s for s, m in zip(result["seeds"], mask, strict=True) if m]


# ------------------------------------------------------------------ console
class Console:
    """A tiny Python console with the library in scope. ``env`` is shared between lines."""

    def __init__(self, get_world, get_sim, get_session=None):
        self._get_world, self._get_sim, self._get_session = get_world, get_sim, get_session
        import lobs

        self.env = {"lobs": lobs, "props": props, "rel": rel, "ops": ops, "act": act, "np": np}

    def namespace(self) -> dict:
        ns = dict(self.env)
        ns["world"] = self._get_world()
        ns["sim"] = self._get_sim()
        if self._get_session is not None:
            ns["session"] = self._get_session()
        return ns

    def run(self, line: str) -> str:
        """Execute one statement or expression; returns the printed output / repr / traceback text."""
        ns = self.namespace()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            try:
                try:
                    code = compile(line, "<console>", "eval")
                    result = eval(code, ns)  # noqa: S307 - the user's own console
                    if result is not None:
                        print(repr(result))
                except SyntaxError:
                    exec(compile(line, "<console>", "exec"), ns)  # noqa: S102
            except Exception:  # noqa: BLE001
                out.write(traceback.format_exc(limit=3))
        for k, v in ns.items():  # keep new names, never the live handles
            if k not in ("world", "sim", "session") and k not in self.env:
                self.env[k] = v
        return out.getvalue()


# ------------------------------------------------------------------ autosave / recovery
def autosave_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p.with_name(f".{p.name}.autosave.yaml")


def write_autosave(session) -> Path | None:
    target = autosave_path(session.path)
    if target is None or not session.dirty:
        return None
    from ...schema.io import save_world

    save_world(session.world.spec, target)
    return target


def recoverable(path) -> Path | None:
    """The autosave of ``path`` if it is newer than the file itself."""
    auto = autosave_path(path)
    if (
        auto is not None
        and auto.exists()
        and Path(path).exists()
        and auto.stat().st_mtime > Path(path).stat().st_mtime
    ):
        return auto
    return None


def clear_autosave(path) -> None:
    auto = autosave_path(path)
    if auto is not None and auto.exists():
        auto.unlink()
