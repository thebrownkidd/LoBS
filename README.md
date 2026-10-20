# The Laws of Bullshit (`lobs`)

A buildable physics world for strange physical situations. You build the world (or import one that
someone else built), define what to measure, and decide what those measurements mean: observations,
rewards, evaluation. **The simulator does not prescribe the task.**

One world / anchor / measurement / experiment API, two independent physics backends
(**MuJoCo** and **Project Chrono**), one engine per run.

```python
import lobs
from lobs import props, rel

world = lobs.World("drop", dt=0.002, substeps=5)                  # one tick = 10 ms
world.add_plane("ground", friction=0.8)
box  = world.add_box("box", size=(0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
top  = box.anchor_on_face("top")                                   # an anchor that moves with the box
goal = world.anchor("goal", pos=(1, 0, 0.1))                       # an anchor fixed in the world

world.measure("height", props.height(box))                         # a property of a thing
gap = world.measure("gap", rel.between(top, goal).distance())      # a relative measure
world.measure("closing", gap.rate())                               # exact derivative, not a finite difference

sim = world.simulate("mujoco", seed=0)                             # or "chrono"
sim.run(seconds=2.0)
print(sim["height"], sim.history["gap"][-5:])
sim.export("runs/drop.npz")                                        # states + measures + metadata
```

## Install

```powershell
winget install prefix-dev.pixi        # once (or see pixi.sh)
pixi install                          # MuJoCo + PyChrono + importers + dev tools, locked
pixi run test                         # unit + cross-engine convention + import tests
pixi run bench                        # physics benchmark scenes on every backend
```

`pip install lobs[mujoco]` gives a MuJoCo-only install (add extras `gltf`, `ifc`, `usd`, `robotics`, `gym`,
`viz`). PyChrono is conda-forge only, hence pixi.

## What you can do

| | |
|---|---|
| **Build** | primitives, articulated trees (hinge / slide / ball / fixed joints), anchors that follow bodies, faces or paths |
| **Measure** | named properties (`lobs.props`) and relative measures between 2-3 things (`lobs.rel`: **distance** and **angle**), composed with operators, `rate()`, `integral()`, `window()`, custom functions. Typed, unit-checked, with provenance |
| **Import** | glTF/GLB, OBJ/STL/PLY, **IFC (BIM)**, **OpenUSD + UsdPhysics**, URDF, SDF, MJCF into a shareable `.lobs` package, with every physics decision logged ([docs/importing.md](docs/importing.md)) |
| **Strange physics** | force laws with measure-valued magnitudes (inverse-cube gravity, wind, springs along frames), time-windowed disturbances |
| **Behaviors** | serializable state machines and `@lobs.behavior` scripts that decide and command (steering, actuators, forces) with latency, period and ownership; triggers (edge, latch, hold, cooldown, chance); triggered, vector-valued, **budgeted adversarial forces** ([docs/behaviors.md](docs/behaviors.md)) |
| **Control / RL** | Gymnasium adapter: you pick observations, reward and termination; initial-state randomization |
| **Compare** | run the same world on MuJoCo and Chrono and get per-measure error, runtime and capability differences |
| **See** | offscreen rendering for either backend, Rerun recordings, world-card previews |
| **Replay** | add anchors and measures to a recorded run and evaluate them without re-simulating |

## Importing someone else's world

```powershell
lobs import campus.ifc -o campus.lobs --profile civil --motion auto   # or: --motion dynamic
lobs check  campus.lobs --backend mujoco                              # stability check; names overlaps, explosions
lobs card   campus.lobs --preview                                     # world card + overview image
lobs import --update campus.lobs                                      # re-run after editing import/overrides.yaml
```

```python
world, report = lobs.import_world("campus.ifc", motion="auto", profile="civil")
for d in report.needs_review:                   # the guesses worth a human look
    print(d.target, d.field, d.value, d.rationale)
sim = world.simulate("mujoco")
```

`motion` is required: it decides what moves when the file does not say. Anything stated by the source, by
profile rules or by your overrides wins over the guess.

## Backends

```python
lobs.backends.available()                     # ['mujoco', 'chrono']
world.simulate("chrono")                      # raises with the exact reason if the engine cannot do something
print(lobs.compare(world, ["mujoco", "chrono"], seconds=4.0))
```

The engines compute dynamics; LoBS defines the conventions (units, frames, joint and actuator laws,
contact conventions) in one place and tests every convention on every backend. Known differences are
declared as *approximations* in `sim.capabilities`, never hidden: see [docs/conventions.md](docs/conventions.md).

## Examples

[examples/](examples/) is a runnable tour (each one is a test): drop a box, a pendulum with a control loop,
self-moving anchors, MuJoCo vs Chrono, replaying with new measures, importing a world, a Gymnasium env,
force laws, rendering and Rerun, and behaviors (a defender state machine and a budgeted gust).

## Layout

```
src/lobs/   schema/ (pure data)  registry kinematics anchors  world.py (builder)  forces.py behavior.py act.py randomize.py
            measure/ (compile, evaluate)  props.py rel.py ops.py   sim.py trajectory.py compare.py check.py
            backends/ (base, capabilities, mujoco_backend, chrono_backend, fake)
            importers/ (readers/{gltf,mesh,ifc,usd,urdf,sdf,mjcf}, normalize, physicalize, geometry_fit, decisions, api)
            gym.py viz.py cli.py
tests/      unit, cross-engine conventions, import fixtures generated at test time
benchmarks/ physics scenes on every backend        docs/   conventions, importing, behaviors
tools/      probe_engines.py (engine versions + API dumps)
```

## Status and honest limitations

Done: the full plan plus behaviors, triggers and adversarial forces. 270+ tests (1 skipped: Chrono has no joint armature), 22 benchmarks. A 20 x 20 city (500 actors) imports in ~8 s and runs at ~2x
real time on MuJoCo (0.2x on Chrono).

Not implemented (and why it matters):
- **Surface clearance** (distance between surfaces, not centres): needs a geometric distance query that is
  not engine-neutral yet.
- **Native MJCF actors** (tendons, equality constraints, muscles): MJCF import is portable; those features are
  reported as "not imported".
- **IFC door hinges / openings** are not modelled: doors import as fixed elements.
- **Deformable / flexible bodies, FEA** (Chrono's headline feature), GPU batching: Phase 3 in `features.md`.
- COLLADA meshes (`.dae`) in URDF/SDF need `pycollada`; `xacro` must be expanded first.
- Chrono is the slower, less exact backend here (hull-based cylinders/capsules, min friction combination, no
  joint armature, impulsive contacts). It is complementary, and the differences are reported.
