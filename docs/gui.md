# LoBS Studio (desktop GUI)

```
pip install "lobs[gui]"        # PySide6 + pyqtgraph
lobs gui [world.yaml | name.lobs]   # or: python -m lobs.gui
```

The GUI is a thin shell over the library: it edits the same `WorldSpec` that `world.save()` writes, and every
action is a library call (`World.validate / simulate`, `Simulation.step / apply_force`, `Trajectory.evaluate`,
`lobs.import_world`, `check_world`, `compare`). Nothing in a saved world depends on the GUI; extra data
(expression texts, state-graph node positions) lives under `metadata.gui` and is ignored by the library.

## Panels

| Panel | What it does |
|---|---|
| **Viewport** | Renders through the engine-independent MuJoCo visual twin, so it looks the same for MuJoCo and Chrono. Left-drag orbits, middle/right-drag or Shift-drag pans, wheel zooms, click selects, **Ctrl-drag pushes a body** with a spring (a live disturbance). `F` frames everything. Overlays: anchors, selection, force arrows (scale in the toolbar), behavior state labels. |
| **Toolbar** | Play/Pause (Space), Step (Right), Reset (Ctrl+R), speed 0.1x-10x or max, backend, seed, force-arrow scale. Editing the world rebuilds the simulation (debounced) and pauses it. |
| **Outline / Inspector** | Tree of actors, bodies, joints, anchors, forces, behaviors and measures. The inspector is generated from the schema (every field of `ActorSpec`, `BodySpec`, `ForceSpec`, ... appears automatically); invalid values are rejected and the world is unchanged. Right-click to duplicate or delete. `Add` menu creates primitives, anchors, forces and adversaries; `Use current state as initial` stores poses and velocities. |
| **Measures / Plots** | Type a measure as Python (`props.height(ball) < 0.5`, `rel.between(a, b).distance()`) with live compile errors and positions; values, units and live plots (vectors split per component), CSV export. Actors and world anchors are in scope by name; `T("path")` for anything else. |
| **Behavior editor** | Node graph of a state machine: drag states, link them, edit `when` / `min_time` / priority, edit `on_enter` / `do` / `on_exit` actions with typed forms, latency, period and actor. **Live debugger**: the current state is highlighted, edges whose condition is true glow, time in state and the transition log are shown, and a breakpoint on a state pauses the run when it is entered. |
| **Forces** | Force laws and adversaries with live impulse bars against their budget, peak and active time; per-law runtime switch; **Inject gust** applies a budgeted push (peak and impulse exactly respected). |
| **Timeline** | Lanes for behavior states and force magnitude over time; click or drag to seek in a replay. |
| **Replay** | Open a saved `.npz`, or replay the last live run; scrub, loop, speed. Evaluate new measures on the recording (live-only measures report the library error). |
| **Import wizard** (`Ctrl+I`, or drop a file on the window) | Choose a foreign file, **choose what moves (auto or dynamic; there is no default)**, run the import in the background, review every decision with provenance and confidence, edit overridable values, re-import, open. |
| **World check / Compare / Monte Carlo** | Stability check, MuJoCo-vs-Chrono table, and N seeded episodes summarized as a histogram; click a bin and replay one of its seeds. |
| **Console** | A Python prompt with `world`, `sim`, `session`, `props`, `rel`, `ops`, `act`, `np`, `lobs` in scope. |
| **Command palette** | `Ctrl+K`, type to run any action. |

Autosave runs every 30 s next to the open file (`.name.yaml.autosave.yaml`) and is offered for recovery when it is
newer than the file; a normal save removes it.

## Trust note
Measure expressions, the console and the Monte Carlo randomize box evaluate the Python you type, in your own
process. Open worlds you trust; a world file itself never executes code (script behaviors are referenced by name
and must be registered in your own Python).

## Architecture (for contributors)
`lobs/gui/core/` has no Qt dependency and holds all the logic: `session` (undo/redo with coalescing, dirty
tracking), `simctl` (time-budgeted stepping), `camera` (projection that mirrors MuJoCo's free camera, verified
against the renderer), `picking`, `forms` (schema introspection), `builder`, `bhv` (state-machine editing),
`exprs`, `plotdata`, `replay`, `importing`, `gust`, `tools`. `lobs/gui/widgets/` are thin Qt views. Tests run with
`QT_QPA_PLATFORM=offscreen` and never touch your saved window layout (`LOBS_GUI_NO_SETTINGS=1`).
