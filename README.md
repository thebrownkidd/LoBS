# The Laws of Bullshit (`lobs`)

A buildable physics world for strange physical situations. You build the world, define what to
measure, and decide what those measurements mean (observations, rewards, evaluation). The simulator
does not prescribe the task.

One world / anchor / measurement / experiment API, two independent physics backends (**MuJoCo** and
**Project Chrono**), one engine per run.

## Install

```powershell
winget install prefix-dev.pixi        # once
pixi install                           # MuJoCo + PyChrono + dev tools, locked
pixi run test                          # unit + convention tests
pixi run bench                         # benchmark scenes on every available backend
```

`pip install lobs[mujoco]` gives a MuJoCo-only install; PyChrono is conda-forge only.

## Five-minute tour

```python
import lobs
from lobs import props, rel

world = lobs.World("drop", dt=0.002, substeps=5)          # tick = 10 ms
world.add_plane("ground", friction=0.8)
box  = world.add_box("box", size=(0.2, 0.2, 0.2), mass=1.0, pos=(0, 0, 1))
top  = box.anchor_on_face("top")                           # an anchor that moves with the box
goal = world.anchor("goal", pos=(1, 0, 0.1))               # an anchor fixed in the world

world.measure("height", props.height(box))                 # a property of a thing
gap = world.measure("gap", rel.between(top, goal).distance())   # a relative measure
world.measure("closing", gap.rate())                       # exact derivative, not a finite difference

sim = world.simulate("mujoco", seed=0)                     # or "chrono"
sim.run(seconds=2.0)
print(sim["height"], sim.history["gap"][-5:])
sim.export("runs/drop.npz")                                # states + measures + metadata
world.save("worlds/drop.yaml")
```

More in [examples/](examples/): articulated control loop, self-moving anchors, engine comparison,
and replaying a recorded run with measures that did not exist when it was simulated.

## Concepts

| | |
|---|---|
| **World** | gravity, timing, contact settings, actors, anchors, measures |
| **Actor** | a named physical thing: static, a rigid body, or an articulated tree (hinge / slide / fixed joints) |
| **Anchor** | a named frame on the world or on a body: rigid, on a face, or moving by a rule (`Path.circle/sine/line/keyframed`, `TranslateOnly`) |
| **Property** (`lobs.props`) | named world-relative quantities of a thing: `height`, `velocity`, `kinetic_energy`, `joint_position`, `contact_force`, ... |
| **Relative measure** (`lobs.rel`) | between 2 or 3 things; the whole vocabulary is **distance** and **angle** |
| **Composition** (`lobs.ops`, operators) | arithmetic, comparisons, `rate()`, `integral()`, `window()`, `duration_true()`, custom functions |

Measures are typed and unit-checked when you define them (`height + speed` is an error), and each
records its provenance (`engine`, `computed`, `estimated`). Every measure is computed from per-body
pose + velocity, so a recorded run can be re-analysed with new anchors and measures without
re-simulating (`Trajectory.evaluate`).

## Backends

```python
lobs.backends.available()                 # ['mujoco', 'chrono']
sim = world.simulate("chrono")            # raises with the exact reason if the world needs something
                                          # the engine cannot do (e.g. triangle-mesh dynamics)
print(sim.capabilities, sim.info)          # approximations are listed, never hidden
cmp = lobs.compare(world, ["mujoco", "chrono"], seconds=4.0)
print(cmp.table())                         # per-measure RMSE / max / final error, runtime, differences
```

The engines compute dynamics; LoBS defines the conventions (units, frames, joint and actuator laws,
contact sign conventions) in one place and tests every convention on every backend
(`tests/backends/test_conventions.py`). Known, documented differences, e.g. Chrono combines friction
with the minimum rule, models cylinders/capsules as convex hulls, and has an impulsive contact model.

## Layout

```
src/lobs/   schema/ (pure data)   registry, kinematics, anchors     world.py (builder)
            measure/ (compile + evaluate)   props.py rel.py ops.py    sim.py trajectory.py compare.py
            backends/ (base, capabilities, mujoco_backend, chrono_backend, fake)
tests/      unit + cross-engine conventions        benchmarks/   physics scenes on every backend
examples/   runnable tour                          tools/probe_engines.py   engine versions + API dumps
```

## Status

Phase 1 (engine-independent core, MuJoCo, Chrono, benchmarks, comparison mode) is complete.
Phase 2: Gymnasium adapter, force laws, mesh / URDF / USD / IFC world import, viewer.
