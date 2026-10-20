# Performance

Throughput is measured with `python benchmarks/bench_throughput.py` (ticks/s, `record=False`, MuJoCo, one process,
16-logical-CPU Windows PC). A tick is `substeps x dt` of physics plus the Python layer (measures, force laws,
behaviors, state read). Raw `mj_step` on the demo world does about 53,075 ticks/s, so the
floor of what the library could reach is that number; the rest is overhead we control.

## Stage A: Python-level speed-ups

| Scene | Before | After | Speed-up |
|---|---:|---:|---:|
| trivial, 3 measures | 6,873 | 14,802 | 2.2x |
| trivial, 0 measures | 13,442 | 14,828 | 1.1x |
| trivial, 20 measures | 5,163 | 14,846 | 2.9x |
| 1 behavior | 2,619 | 4,982 | 1.9x |
| 30 bodies | 1,872 | 2,824 | 1.5x |
| demo world | 2,530 | 5,274 | 2.1x |
| zoo (many measures) | 374 | 3,338 | 8.9x |
| behaviors scenario | 1,001 | 2,462 | 2.5x |
| gym step | 6,365 | 8,616 | 1.4x |

What changed (all results identical to the recorded golden trajectories, `tests/test_golden.py`, to 1e-9; the one
accepted difference is that rigid-anchor velocities are now analytic instead of a +-1 us finite difference, which
moved `anchor.velocity` by 4e-10):
- **Lazy measures.** Side-effecting nodes (stateful operators, random draws, live runtime reads) still run every
  tick in node order, so semantics and random streams are unchanged; every other measure is computed only when
  something reads it (an observation, a reward, an active behavior action, the history when recording). With
  `record=True` all measures are read, so recording costs what it did before.
- **One `EvalCtx` per tick** shared by measures, behaviors and force laws; cached body COMs; single-quaternion fast
  paths (no `np.linalg.norm`/`np.cross`, which cost 8-60 us on 3-vectors); analytic anchor velocity.
- **MuJoCo backend.** One `mj_step(nstep)` call when no wrenches are applied, same-body wrenches merged, and the
  state read is vectorised (no per-body `mj_objectVelocity`).
- **Gym.** `frame_skip` (agent decision interval) and `terminate_each_tick`; `lobs.make_vec(world, n, ...)` builds a
  Gymnasium vector env with one process per environment. On this PC the demo world scales to about 23k ticks/s in
  aggregate at 8 workers and stays flat beyond that, because the parent process and the physical core count become
  the limit. Use a larger `frame_skip` (10-20) so each worker step is long compared with the pipe round trip.

What did not change: a world with many contacts is bound by MuJoCo itself (the 30-body scene runs within 15% of raw
MuJoCo), and the per-tick Python cost of behaviors and force laws (a few hundred microseconds) is the next target.
