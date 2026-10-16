# Conventions (the cross-engine contract)

Every item below is pinned by a test that runs on every available backend
(`tests/backends/test_conventions.py`, `tests/backends/test_ball_joint.py`, `benchmarks/`).

## Units, frames, rotations
- SI units, angles in radians. Degree inputs use explicitly named parameters.
- World frame: right-handed, **Z up**. Default gravity `(0, 0, -9.81)`.
- Quaternions are `(w, x, y, z)`, Hamilton product, double-cover aware (`q` and `-q` are the same rotation).
- A body's frame is its authored **reference frame**, not its centre of mass. Backends report the pose and
  twist of the reference frame (velocity of the reference *origin*, world angular velocity, both in world axes).
  The centre of mass is stored separately (`inertial.com`).

## Shapes (local z axis)
Box `size` = **full** extents. Sphere `radius`. Cylinder `radius`, `length` (full height). Capsule `radius`,
`length` (cylindrical segment only). Plane: local `z = 0`, normal +z, infinite, fixed bodies only.
Convex meshes / sets use convex hulls; triangle meshes are **fixed bodies only**.

## Joints (tree structures; one joint per child body)
| type | `q` | `qd` | notes |
|---|---|---|---|
| hinge | right-hand rotation about the axis (rad), **unwrapped** | rad/s | axis/pos in the child frame |
| slide | translation along the axis (m) | m/s | |
| ball | rotation vector of the child relative to its zero pose (3 values, rad) | relative angular velocity **in the child frame** (3 values) | no limits; no actuators |
| fixed | - | - | |

`q = 0` is the pose the child has when authored relative to its parent. Joint arrays (`joint_q`,
`joint_qd`, `joint_effort`) have one slot per DOF (a ball joint occupies three).

## Actuators (identical on every backend)
Clip the control to `ctrl_range`, then apply the law, then clip the effort to `effort_limit`:
`effort`: `tau = u`; `position`: `tau = kp (u - q) - kd qd`; `velocity`: `tau = kd (u - qd)`.
`joint_effort` is the effort applied during the **last substep**.

## Forces and time
- One tick = `substeps x dt`. Actions, measures, history and force laws run on ticks; physics substeps are internal.
- External wrenches and force laws are applied in the world frame and held for all substeps of the tick
  (zero-order hold). Force-law magnitudes are evaluated from the state at the start of the tick.
- `t = 0` at reset; reset clears all histories and stateful measures, so there are never stale values.

## Contacts
Rows are ordered `body_a < body_b`; `normal` is a world unit vector from A to B; `depth >= 0` means penetration;
`force_on_b` is the world-frame force on B; `normal_force >= 0`. MuJoCo reports the end-of-tick state; Chrono
averages force over the substeps of a tick (its contact force alternates between ~0 and ~mg for resting bodies).
Contact recording in trajectories is opt-in (`record_contacts=True`); contact *measures* work regardless.

## Materials and filtering
Friction between two shapes: MuJoCo uses `max(mu_a, mu_b)`; Chrono's default composition is the minimum, which
cannot be overridden from Python, so Chrono declares `FRICTION` approximate (equal coefficients are exact).
Collision groups 0-15 with a list of disabled group pairs; **parent/child bodies never collide** (MuJoCo only
does this by default when the parent is not welded to the world, so the backend excludes the pair explicitly).

## Accelerations
Coordinate acceleration (free fall reads `(0, 0, -9.81)`, rest reads 0). MuJoCo: engine; Chrono: backward
difference of velocity (declared approximate).

## Static vs dynamic
A body is static if its actor root is fixed and every joint between it and the root is `fixed`. Static bodies
are not in the per-tick state arrays; their world pose is a registry constant.

## Measurements
- Rates (`m.rate()`) are exact: they evaluate the measure at the state advanced along the engine twists by
  `+-1e-6 s`. They do not depend on `dt` and do not smear at impacts. Only pose-like measures allow `rate()`;
  others use `derivative()` (a finite difference, marked `estimated`).
- Angles use `atan2` formulations; degenerate geometry (coincident points, zero projection) gives **NaN**, never 0.
- Every measure records its unit, shape, kind and provenance (`engine`, `computed`, `estimated`, `model`).

## Backend capability differences (declared in `sim.capabilities`)
| | MuJoCo | Chrono |
|---|---|---|
| cylinder, capsule | native | convex hull (approximate) |
| plane | native | large box (approximate) |
| convex mesh / set | hull collision (approximate) | convex hulls |
| triangle mesh | unsupported | static only |
| restitution | emulated via contact `solref` | NSC restitution (approximate) |
| friction combination | max | min (approximate) |
| joint armature | native | unsupported |
| body acceleration | native | backward difference |
| contact forces | end of tick | averaged over substeps |
