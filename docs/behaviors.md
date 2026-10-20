# Behaviors, triggers and adversarial forces

## Behavior vs force
A **force law** (`world.add_force`) is a stateless function of the state: `state -> force`.
A **behavior** (`world.add_behavior`) owns state (a state machine, variables, a seeded RNG), decides, and
*writes commands*: actuator targets, forces (including `Steer`, acceleration-limited locomotion), and
switches for force laws and other behaviors. A behavior can therefore produce forces, but it is not one.

## Tick order
```
measures evaluated on the state -> external action sets actuator targets -> behaviors run
(commands win over the external action for actuators they own) -> force laws (when / budget / switches)
-> wrenches -> backend.step(substeps) -> read state
```
All behaviors read the same tick-start snapshot (optionally `latency` old), so the order in which behaviors
are declared never changes results. `EnableForce` / `EnableBehavior` take effect after every behavior has
decided. Nothing here touches the backends: only `add_wrench` and actuator targets are used.

## Triggers are boolean measures
They work in force laws (`when=`), transitions, Gym rewards and plots, and serialize with the world.

| | |
|---|---|
| `m.rising()` / `m.falling()` | one-tick pulse on an edge (a `True` on the first tick counts as rising) |
| `ops.latch(set_, reset)` | SR latch, initially False, **reset wins** |
| `m.hold(s)` | stays true for `s` seconds after the last true (re-triggerable); debounce / grace period |
| `m.cooldown(s)` | pulse on an accepted rising edge, ignores edges for `s` seconds |
| `m.once()` | pulse on the first rising edge after reset |
| `ops.chance(p)` | Bernoulli(p) per tick from the run RNG; gate with `.rising()` for per-event odds |
| `ops.uniform / normal` | random numbers, resampled every tick; `value.sample_on(trigger)` samples once per trigger |
| `ops.event(name)` | true for the tick after a behavior `Emit(name)` |
| `props.behavior_state / behavior_in_state / behavior_time_in_state / behavior_var` | read behavior state |
| `ops.vec / unit / clip_norm / as_unit` | vector helpers (`unit` of a zero vector is zero, never NaN); `as_unit` gives gains their units |

Cookbook: *debounce* `cond.hold(0.2)`; *one-shot* `cond.once()`; *random delay*
`ops.uniform(0.1, 0.4).sample_on(trigger)`; *latch* `ops.latch(near.rising(), far.rising())`.

## State machines
```python
sm = lobs.StateMachine("guard_then_chase")
sm.var("alert", 0.0)
sm.state("guard", do=[act.steer(to=zone, mode="arrive", speed=2.0, max_accel=6.0)])
sm.state("chase", on_enter=[act.emit("alerted")],
         do=[act.steer(to=ball, mode="intercept", speed=5.0, max_accel=9.0)])
sm.transition("guard", "chase", when=rel.between(me, ball).distance() < 4.0, min_time=0.2)
world.add_behavior("defender", on=me, spec=sm, latency=0.15, period=0.05)
```
* Transitions from the current state are tried by `priority` (higher first), then declaration order; at most
  one fires per decision; `min_time` is the time already spent in the state.
* Actions: `act.actuate`, `act.force`, `act.steer`, `act.set_var`, `act.emit`, `act.enable_force`,
  `act.enable_behavior`. Values and conditions are measures. `on_enter` / `on_exit` actions run once on the
  decision where the transition fires; `do` actions are held until the next decision (`period`).
* `Steer` modes: `seek`, `arrive`, `flee`, `intercept` (leads by distance / speed), `hold` (brake).
  The force is `m * clip_norm(gain * (v_des - v), max_accel)`, planar by default. It is honest physics:
  acceleration-limited and opposed by friction. Jumping or kicking is `act.force` behind a trigger.
* Hidden measures (`__beh.*`) of **all** states evaluate every tick so stateful nodes keep updating; only the
  values of active actions are used. The cost grows with the number of states.
* Fairness knobs: `latency` (sensing delay, exact in ticks), `period` (decision interval), force `budget`.

## Python behaviors
```python
@lobs.behavior("wander")
def wander(ctx):                       # or a class with reset(ctx) / step(ctx)
    ctx.force("me", (ctx.rng.uniform(-1, 1) * ctx.params["scale"], 0, 0))

world.add_behavior("w", on=me, script="wander", params={"scale": 2.0}, actuates=["robot/hip"])
```
`ctx` offers `t dt tick rng params state`, `measure(name)` (latency-delayed), `pos/vel(thing)`, and `steer`,
`actuate`, `force`, `emit`, `enable_force`, `enable_behavior`. `ctx.rng` is a per-behavior seeded RNG, so runs
are reproducible per `seed`. A script must list in `actuates=` every actuator it commands (ownership).
`lobs.behavior.policy("name", fn, observe=[...], actuate=[...])` registers an `obs -> action` callable (a
scripted or learned opponent) the same way. Scripts are referenced by name, so a world file is only runnable
where the name is registered; validation reports unregistered scripts.

## Ownership, Gym, randomization
Actuators a behavior commands are *owned*: external actions for them are ignored (one warning for dict
actions), two owners are a validation error, and `GymEnv` removes them from the action space.
`info["behaviors"]` reports state-machine states.
`reset(randomize={"behaviors": {"defender": {"params": {...}, "variables": {...}}}})` (with `Uniform` etc.)
overrides script params / machine variables per episode (opponent-difficulty curricula).

## Adversarial forces
`world.add_adversary(name, on, budget=lobs.Budget(peak=..., impulse=..., active_time=...), vector=..., when=...)`
is `add_force(adversary=True)` with a required `peak`. Budgets are per episode: `peak` clamps |F|, `impulse`
caps the total `integral |F| dt` (the last tick is scaled so it is exhausted exactly), `active_time` caps the
active seconds. Read them with `sim.force_stats[name]` or `props.force_impulse / force_active_time /
force_active / force_applied`.
* *Budgeted gust*: `vector=ops.vec(0, 2, 0)`, `when=` a time window, `Budget(peak=2, impulse=1)`.
* *Reactive shove*: `vector=ops.unit(pos(ball) - pos(goal)) * 3`, `when=` agent inside a region.
* *Opposing drag*: `vector=props.velocity(agent) * -k` (give it units with `ops.as_unit(..., "newton")`).

## Determinism and replay
Everything draws from seeded RNGs; two runs with the same seed are bitwise identical on MuJoCo. Random and
behavior-state measures are live-only: `Trajectory.evaluate` / `rate()` reject them loudly instead of
returning stale values. Trajectories record the applied force of each law and the state of each behavior;
`lobs.viz.log_rerun` draws force arrows and logs behavior transitions.

## Recipe: grip a ball without a grip feature
```python
near = rel.between(ball, grip).distance() < 0.35
holding = ops.latch(near.rising(), (dist > 0.5) | (ops.time() >= 1.5))
spring = (ops.as_unit((pos(grip) - pos(ball)) * 400, "newton")
          + ops.as_unit((vel(hand) - vel(ball)) * 25, "newton"))
world.add_force("grip", on=ball, vector=spring, when=holding, budget=lobs.Budget(peak=200))
```
Caught, held against gravity, released on a trigger, knocked free by a disturbance larger than the peak
(`tests/test_behaviors.py::test_grip_recipe_from_primitives`).
