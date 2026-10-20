"""Throughput benchmark: ticks/s of the Python layer vs raw MuJoCo.

    pixi run python benchmarks/bench_throughput.py [--json out.json]
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lobs  # noqa: E402
from lobs import act, props  # noqa: E402
from tests.golden.scenarios import SCENARIOS  # noqa: E402


def rate(fn, n, warm=200):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return n / (time.perf_counter() - t)


def ball_world(measures=3, behavior=False, bodies=1):
    w = lobs.World("p", dt=0.002, substeps=5)
    w.add_box("floor", (20, 20, 0.2), fixed=True, pos=(0, 0, -0.1))
    for i in range(bodies):
        w.add_sphere(f"b{i}", 0.1, mass=0.5, pos=(i * 0.5, 0, 1.0), restitution=0.8)
    for k in range(measures):
        w.measure(f"m{k}", props.position("b0").z + k)
    if behavior:
        sm = lobs.StateMachine("s")
        sm.state("a", do=[act.steer(to="b0", mode="seek", speed=1.0)])
        w.add_behavior("bh", on="b1", spec=sm)
    return w


def sim_rate(world, n=2000):
    s = world.simulate("mujoco", record=False)
    r = rate(s.step, n)
    s.close()
    return r


def raw_rate(world, n=3000):
    import mujoco

    s = world.simulate("mujoco", record=False)
    m, d = s.native
    r = rate(lambda: mujoco.mj_step(m, d, 5), n)
    s.close()
    return r


def gym_rate(n=2000):
    env = lobs.GymEnv(ball_world(), "mujoco", observe=["m0", "m1"], reward="m2", max_seconds=2.0)
    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape)

    def step():
        _, _, te, tr, _ = env.step(zero)
        if te or tr:
            env.reset()

    r = rate(step, n)
    env.close()
    return r


def main() -> dict:
    out = {}
    out["trivial, 3 measures"] = sim_rate(ball_world())
    out["trivial, 0 measures"] = sim_rate(ball_world(0))
    out["trivial, 20 measures"] = sim_rate(ball_world(20))
    out["1 behavior"] = sim_rate(ball_world(behavior=True, bodies=2), 1000)
    out["30 bodies"] = sim_rate(ball_world(bodies=30), 1000)
    out["demo world"] = sim_rate(SCENARIOS["demo"]()[0])
    out["demo world (raw mujoco)"] = raw_rate(SCENARIOS["demo"]()[0])
    out["zoo (many measures)"] = sim_rate(SCENARIOS["zoo"]()[0], 1000)
    out["behaviors scenario"] = sim_rate(SCENARIOS["behaviors"]()[0], 1000)
    out["gym step"] = gym_rate()
    return out


if __name__ == "__main__":
    res = main()
    width = max(len(k) for k in res)
    for k, v in res.items():
        print(f"{k:<{width}}  {v:>10,.0f} ticks/s")
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(res, indent=2))
