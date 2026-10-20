# Performance

Throughput is measured with `python benchmarks/bench_throughput.py` (ticks/s, `record=False`, MuJoCo, one process,
16-logical-CPU Windows PC). A tick is `substeps x dt` of physics plus the Python layer (measures, force laws,
behaviors, state read). Raw `mj_step` on the demo world does about 53,075 ticks/s, so the
floor of what the library could reach is that number; the rest is overhead we control.

## Results (ticks/s, one process)

| Scene | Before | After Stage A (Python) | After Stage B (native) | Total |
|---|---:|---:|---:|---:|
| trivial, 3 measures | 6,873 | 14,802 | 26,131 | 3.8x |
| trivial, 0 measures | 13,442 | 14,828 | 24,826 | 1.8x |
| trivial, 20 measures | 5,163 | 14,846 | 25,266 | 4.9x |
| 1 behavior | 2,619 | 4,982 | 19,023 | 7.3x |
| 30 bodies | 1,872 | 2,824 | 3,044 | 1.6x |
| demo world | 2,530 | 5,274 | 26,234 | 10.4x |
| zoo (many measures) | 374 | 3,338 | 4,280 | 11.5x |
| behaviors scenario | 1,001 | 2,462 | 16,895 | 16.9x |
| gym step | 6,365 | 8,616 | 16,521 | 2.6x |

Raw `mj_step` on the demo world is 53,706 ticks/s, so the demo now runs at about half of what
the engine alone could do. Multi-process (`lobs.make_vec`, 16 workers, `frame_skip=20`, demo world): about
**106,000 ticks/s in aggregate** (23,000 before Stage B).

The scenes that stayed slow are bound by something else: `30 bodies` is MuJoCo itself (contact heavy), and `zoo`
reads contacts and accelerations every tick, which still goes through the Python state read.

## Stage A: Python-level speed-ups

(see the combined table below)

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

## Stage B: the native tick core

`src/native/` is a C++17 extension (nanobind) that runs one whole tick in a single call: behaviors, force laws,
wrench application, the MuJoCo step, the state read and the measure program. It is **optional**: when it is not
built (or `LOBS_NATIVE=0`), everything runs on the pure-Python reference implementation, which stays the
specification (the whole test suite passes on both paths).

Build (Windows needs the MSVC Build Tools; the pixi environment provides cmake, ninja and nanobind):

```
pixi run python tools/build_native.py      # downloads the MuJoCo headers for the installed version once
```

What runs natively: the measure interpreter (arithmetic, comparisons, logic, vector helpers, every trigger and
time-dependent operator, body / actor / rigid-anchor / joint properties, `rel` distance, behavior / force /
event measures), state-machine behaviors (priority, `min_time`, `on_enter` / `on_exit`, variables, events,
latency, period, ownership, switches, `Steer`) and force laws (`when`, vector, budgets, reaction, toward / direction
/ axis). Any measure it cannot express (angles, rates, contact and acceleration properties, custom functions,
random draws, `props` that need inertia, path-rule anchors) is evaluated in Python from the native values.

A simulation uses the native core only when all of this holds, otherwise it stays on the Python path (reported in
`sim.info["native_core"]`): MuJoCo backend without ball joints, no script behaviors, no contact or acceleration
measures, `record_contacts` off. Chrono always uses the Python path.

Equivalence: measure histories, force statistics and behavior state agree with the Python path to 1e-9 on the golden
scenarios and on dedicated parity worlds (`tests/native/`); transition logs and event times are identical, and two
native runs are bitwise identical. Differences are last-bit effects of scalar C++ arithmetic versus numpy/BLAS.
