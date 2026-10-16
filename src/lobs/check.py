"""Headless stability check for a world (used by `lobs check` and the world card)."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

PENETRATION_LIMIT = (
    0.02  # m: approximated collision hulls (CoACD) are not exact; real overlaps are cm-m deep
)
EXPLOSION_SPEED = 20.0  # m/s
FALL_LIMIT = 10.0  # m below the lowest initial body


@dataclass
class CheckResult:
    backend: str
    engine_version: str
    seconds: float
    passed: bool
    realtime_factor: float
    contacts: int
    problems: list = field(default_factory=list)
    initial_penetrations: list = field(default_factory=list)  # [(pair, depth)]
    warnings: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["initial_penetrations"] = [[list(p), float(x)] for p, x in self.initial_penetrations[:20]]
        return d

    def summary(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'} [{self.backend} {self.engine_version}] {self.seconds:g}s simulated, " \
               f"real-time factor {self.realtime_factor:.1f}x, {self.contacts} contacts at the end"  # fmt: skip
        return "\n".join(
            [head, *[f"  - {p}" for p in self.problems], *[f"  ! {w}" for w in self.warnings]]
        )


def check_world(
    world,
    backend: str = "mujoco",
    seconds: float = 2.0,
    penetration_limit: float = PENETRATION_LIMIT,
) -> CheckResult:
    """Simulate with no actions; report initial penetrations, NaN, explosions, falling out of the world."""
    report = world.validate(backend)
    if not report.ok:
        from .backends import get

        try:
            version = get(backend).engine_version()
        except Exception:  # noqa: BLE001
            version = "?"
        return CheckResult(backend, version, 0.0, False, 0.0, 0, [i.message for i in report.errors])
    sim = world.simulate(backend, seed=0, record=False)
    problems, penetrations = [], []
    reg = sim.reg
    f0 = sim.backend.read_state(
        contacts=True, accelerations=False
    )  # contacts are not read when not recording
    if f0.contacts is not None and len(f0.contacts):
        deep = f0.contacts.depth > penetration_limit
        seen = set()
        for a, b, d in zip(
            f0.contacts.body_a[deep], f0.contacts.body_b[deep], f0.contacts.depth[deep], strict=True
        ):
            key = (reg.bodies[int(a)].path, reg.bodies[int(b)].path)
            if key not in seen:
                seen.add(key)
                penetrations.append((key, float(d)))
    if penetrations:
        worst = max(penetrations, key=lambda x: x[1])
        problems.append(f"{len(penetrations)} body pairs start interpenetrating by more than {penetration_limit * 1e3:g} mm (worst {worst[0]} at {worst[1] * 1e3:.1f} mm)")  # fmt: skip
    floor = float(f0.body_pos[:, 2].min()) if len(f0.body_pos) else 0.0
    t0 = time.perf_counter()
    nonfinite = exploded = fell = False
    culprit = "?"
    ticks = int(round(seconds / sim.tick_dt))
    for _ in range(ticks):
        sim.step()
        f = sim.state()
        arrays = (f.body_pos, f.body_quat, f.body_linvel, f.body_angvel, f.joint_q, f.joint_qd)
        if any(not np.all(np.isfinite(a)) for a in arrays):
            nonfinite = True
            break
        if len(f.body_linvel):
            speeds = np.linalg.norm(f.body_linvel, axis=1)
            if speeds.max() > EXPLOSION_SPEED and not exploded:
                exploded = True
                culprit = reg.bodies[reg.dynamic_order[int(speeds.argmax())]].path
        if len(f.body_pos) and f.body_pos[:, 2].min() < floor - FALL_LIMIT:
            fell = True
    wall = max(time.perf_counter() - t0, 1e-9)
    if nonfinite:
        problems.append("simulation produced NaN/inf")
    if exploded:
        problems.append(
            f"body {culprit} exceeded {EXPLOSION_SPEED:g} m/s (unstable contact or constraint, often an initial overlap)"
        )
    if fell:
        problems.append(
            f"a body fell more than {FALL_LIMIT:g} m below the scene (missing ground or collision?)"
        )
    n_contacts = len(sim.state().contacts) if sim.state().contacts is not None else 0
    res = CheckResult(backend, str(sim.info.get("version", "?")), sim.time, not problems, sim.time / wall, n_contacts,
                      problems, penetrations, [i.message for i in report.warnings])  # fmt: skip
    sim.close()
    return res
