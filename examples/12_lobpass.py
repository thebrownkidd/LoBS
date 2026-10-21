"""The lob-pass demo in one play: a scripted passer finds the dunking lob, the dunker runs his path and finishes it.

A passer throws a lob to a running dunker. Here the throw comes from the demo's scripted oracle (the trained
policy is in demos/lobpass/README.md); everything else is the real LoBS world: a drag-affected ball held by a
spring force, players on slide joints driven by state machines, and event measures that decide the outcome.
"""

import numpy as np

from demos.lobpass import constants as C
from demos.lobpass import oracle
from demos.lobpass.env import LobPassEnv
from demos.lobpass.scenario import sample_scenario

env = LobPassEnv({"S0": 1.0})
scn = sample_scenario("S0", 7)
sol = oracle.solve(env, scn)
print(f"dunker starts at {np.round(scn.dunker_start, 1)}, takes off at {np.round(scn.takeoff, 1)}")
env.reset(options={"scenario": scn})
for _ in range((sol["release"] - C.WINDUP_TICKS) // C.DECISION_TICKS):
    env.step(np.array([-1.0, 0.0, 0.0, 0.0]))
_, reward, _, _, info = env.step(sol["action"])
print(f"released at {info['t_release']:.2f} s with v0 = {np.round(info['v0'], 1)} m/s -> {info['outcome']} (reward {reward:+.1f})")
assert info["outcome"] == "DUNK"
