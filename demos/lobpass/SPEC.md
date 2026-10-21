# Part A — Simulation specification (frozen)

### A1 Frames, units, timing
- SI, radians, Z-up, right-handed, quaternions (w,x,y,z), per `docs/conventions.md`.
- `dt=0.002`, `substeps=5`, so **tick = 0.01 s**.
- **Court frame:** origin on the floor directly under the rim center. +x points toward half court. +y is to the right of
  someone standing at half court facing the basket.
- Three.js (Y-up) mapping: `three = (x, z, -y)`, a proper rotation. Apply it in one function only, `player/scene.js::toThree`.

### A2 Court and hoop (NBA)
| Item | Value |
|---|---|
| Rim top height | 3.048 m; tube radius 0.008; inner radius 0.2286; ring centerline radius 0.2366; tube center z 3.040 |
| Rim center | (0, 0) |
| Backboard | 1.829 (y) × 1.067 (z) × 0.05 (x); front face x = −0.381; bottom edge z 2.900; center (−0.406, 0, 3.4335) |
| Rim bracket | box 0.152 × 0.05 × 0.03 centered (−0.305, 0, 3.030) |
| Baseline / half-court / sidelines | x = −1.600 / x = 12.726 / y = ±7.620 |
| Lane | \|y\| ≤ 2.438 from baseline to free-throw line x = 4.191; FT circle r 1.829 at (4.191, 0) |
| Three-point | arc r 7.239 about rim center where \|y\| ≤ 6.706; corner lines \|y\| = 6.706 for x ≤ 2.726 |
| Restricted arc | r 1.219; line width 0.0508 everywhere |
| Stanchion | visual only, base at x = −2.40 |

LoBS bodies:
- `floor`: plane, friction 0.8.
- `backboard`: fixed box, restitution 0.6.
- `rim`: fixed actor, one body of **16 capsules**, radius 0.008, straight length `2·0.2366·sin(π/16)`, tangent to the ring at z 3.040.
- `bracket`: fixed box.

All of these are in collision group 0.

### A3 Ball and flight
- Sphere: r 0.1193, m 0.62, restitution 0.78, friction 0.6, collision group 1 (collides with group 0 only).
- **Drag:** `F = −C·|v|·v` with `C = 0.5·1.204·0.50·π·0.1193² = 0.013459 kg/m`. This is a force law
  `vector=as_unit(V(ball)*N(speed(ball))*(−C), "newton")`. No Magnus effect; spin is visual only.
- **Hold point:** `hold = passer_xy + 0.25·û_rim + (0,0,1.25·H_p)`, where `û_rim` points from the passer to the rim.
  The passer has a trivial state machine `passer` (one state, no actions) whose variable `hold_offset` (3-vec) stores
  `0.25·û_rim + (0,0,1.25·H_p)`.
- **Hold law** (`when=~released`):
  `F = 400·(hold − p_ball) − 25·v_ball + (0, 0, 0.62·9.81)` N, where `released = latch(N(speed(ball)) > 2.0, time < −1)`.
- **Throw:** at the release tick the env calls `sim.apply_force("ball", 0.62·(v0 − v_ball)/0.01, ticks=1)`. During that
  tick the hold law cancels gravity. After it, the velocity is v0 to within 1e-3 m/s (test L1). Next tick `released`
  latches and the hold turns off.
- **Wind-up:** release happens 15 ticks (0.15 s) after the throw decision. The decision is locked during wind-up.
- **Speed cap:** \|v0\| ≤ 14.0 m/s.

### A4 Bodies (anthropometrics)
- Standing reach `R_s = 1.33·H`. Jump: `vz0 = sqrt(2·9.81·L)`, apex time `t_ap = vz0/9.81`, airtime `2·t_ap`.

| | S0 | S1+ |
|---|---|---|
| Passer H | 1.91 | U(1.78, 2.06) |
| Dunker H / L | 1.98 / 0.80 | U(1.85, 2.16) / U(0.65, 0.95), then `L = max(L, 3.35 − 1.33·H)` |
| Defender H / L | — | U(1.80, 2.20) / U(0.50, 0.85) |

- **Rig per player** (actor, `root="fixed"`): `base` → `sx` (Slide x) → `sy` (Slide y) → `body` (Slide z, range (0, 1.5)).
  - `base`, `sx` and `sy` carry a sphere r 0.02, mass 0.1, group 3.
  - `body` has its frame origin at floor level and carries one capsule: radius 0.24, straight length 1.52, centered at
    z 1.0 (a fixed 2.0 m proxy). Mass 90, group 2, friction 0.2.
  - Per-episode heights exist only in behavior variables and rules; LoBS geometry is a proxy.
  - `P(body) = (x, y, jz)`.
  - Disabled collision pairs: (1,2) (2,0) (3,·). Enabled: (2,2), so players physically can't overlap.
- **Jump:** an `on_enter` force for one tick: `F = 90·((vh_des − v_xy)/0.01 + (0, 0, vz0/0.01 + 9.81))`, where `vz0 = bodies.jump_cmd_vz(L)` solves `vz0²/2g + vz0·0.005 = L` (the impulse tick itself lifts the body `vz0·0.005`), so the apex is exactly L. Landing impact may dip jz to −4 cm for a few ticks; the animation clamps jz at 0.
  Test L1: apex within ±1 cm of L. Grounded jz must stay in [−0.005, 0.002].
- Inactive defender slots: `role=0`, parked at `jx = 40 + 3i`, `jy = 40`. Every rule multiplies by `present_i`.

### A5 Scenarios per stage (`scenario.py`)
- **Passer:** polar about the rim, r U(5.5, 8.0), θ U(−75°, 75°) measured from +x. Rejection-sample until inside bounds
  shrunk by 0.4 m.
- **Dunker start:** r U(6.0, 10.0), θ U(−85°, 85°), with \|θ_d − θ_p\| ≥ 35° and ≥ 3.0 m from the passer. He starts
  standing. `start_delay` U(0.0, 0.4) s.
- **Speeds:**
  - `vmax` U(5.0, 7.0)
  - `v_corner` = vmax·U(0.55, 0.75)
  - `v_gather` U(3.0, 4.0)
  - `jump_vh = min(v_gather, (r_to − 0.30)/t_ap)`, aimed at the rim center
- **Takeoff point T:** `r_to` U(1.0, 1.6); angle = approach angle + U(−25°, 25°), clamped to ±70°.
- **S0/S1 path style:** 50% straight, 50% one corner. The corner C sits at fraction U(0.35, 0.65) along start→T, offset
  laterally ±U(1.5, 3.5). It requires a turn of 25–75°, must be in bounds, and ≥ 1.5 m from the rim.
- **S2+:** the planner (A6) chooses the path.
- **Defenders:**

| Stage | n | Roles | Speed | React (s) |
|---|---|---|---|---|
| S0, S1 | 0 | — | — | — |
| S2 | U{1..4} | STATIC (arms up, never moves or jumps) | — | — |
| S3a | U{1..4} | JUMPER (feet fixed, reads ball, jumps in place) | — | U(0.25, 0.35) |
| S3b | U{1..3} | ONBALL ≤1, CHASER, CUTOFF, RIM ≤1 | U(4.5, 6.0)×0.8 | 0.30 |
| S3c | U{2..5} | ONBALL ≤1, CHASER, CUTOFF, RIM ≤2 | U(4.5, 6.0)×1.0 | U(0.15, 0.25) |

- **Placement (S2/S3a):** mixture of passing lane 0.35 (segment passer→T at fraction U(0.3, 0.7), lateral N(0, 0.5)),
  rim zone 0.25 (r U(0.8, 2.5), angle ±70°), dunker straight line 0.25, and on-ball 0.15 (≤1; 1.1 m from the passer toward
  the rim).
- **Role starts (S3b/c):**
  - ONBALL as above
  - CHASER: 1.5–2.5 m from the dunker start
  - CUTOFF: planned path at fraction U(0.4, 0.8), offset U(2, 4) m to the help side
  - RIM: r U(1.0, 2.0) at ±60°
- Constraints for all defenders: ≥ 1.0 m apart, ≥ 2.0 m from the dunker start, ≥ 1.0 m from the passer, in bounds.
  Sampling retries at most 50 times, then resamples the whole scenario.

### A6 Dunker planner (`planner.py`, S2+, deterministic)
- **Candidates:**
  - Takeoff points T: `r_to ∈ {1.0, 1.3, 1.6}` × angle ∈ {−70°, −60°, …, 70°}.
  - Corners: none, plus a 0.75 m grid inside the start/T bounding box grown by 3 m, in bounds and ≥ 1.5 m from the rim.
  - Keep a candidate only if turn ≤ 75° and length ≤ 1.5·\|start − T\|.
- **Cost:** `J = L_path/vmax + 1.5·Σ_i max(0, 1.8 − c_i)² + 0.3·turn/90°`, where `c_i` is the min distance from the
  polyline to defender i's start. Take the argmin, breaking ties by enumeration order. Vectorized; ≤ 5 ms per scenario.

### A7 State machines (`machines.py`, native ops only; `N(x)=as_unit(x,"dimensionless")`, `flat=vec(1,1,0)`)
**Dunker `dunker`** (`period=0.01`, `latency=0`).
- Variables: `corner, takeoff` (3), `has_corner, vmax, v_corner, v_gather, start_delay, jump_vz, jump_vh, H, R_s, L, present=1`.
- Move law: `F = 90·clip_norm(4.0·(û(target − p)_flat·s_des − v_flat), 7.0)` newtons.

| State | Action | Exit |
|---|---|---|
| wait | `steer(mode="hold")` | t ≥ start_delay → run_c (has_corner) / run_t |
| run_c | move law, target C, `s_des = min(vmax, v_corner + 1.5·max(d_C − 0.6, 0))` | d_C < 0.6 → run_t |
| run_t | move law, target T, `s_des = min(vmax, v_gather + 1.5·max(d_T − 2.0, 0))` | d_T < 0.25 (min_time 0.05) → jump |
| jump | on_enter: jump impulse (A4) with `vh_des = jump_vh·û(rim − p)` | always (min_time 0.01) → air |
| air | none | jz < 0.003 and time in state > 0.1 → land |
| land / blocked | `steer(mode="hold")` | — |

`blocked` is entered from wait, run_c or run_t when the `blocked` measure (A8) is true, at priority 10.

**Defender `def{i}`** (`period=0.05`, `latency=0.05`).
- Variables: `present, role (0 off, 1 static, 2 jumper, 3 onball, 4 chaser, 5 cutoff, 6 rim), home (3), speed, R_s, L,
  vz0, react, jump_ready=1`.
- `seen = released & (time − time.sample_on(released) ≥ react)`.
- **Ball read (drag-free, as a human would read it):**
  - `z_r = R_s + L − 0.05`
  - `disc = u_z² + 2g(b_z − z_r)`
  - `τ = where(b_z > z_r, (u_z + max(disc, 0)^0.5)/g, 0)`
  - `aim_ball = b_xy + u_xy·where(b_z > z_r, τ, 0.15)`
- **Jump trigger:** `seen & jump_ready & jz < 0.01 & |aim_ball − p|_xy ≤ 0.45 & τ ≤ vz0/g + 0.03`.
- **Dunk-contest trigger** (roles 5 and 6): dunker airborne (jz > 0.1), dunker vz > 0, and \|p_dk − p\|_xy ≤ 1.2.
- **Rim-front of the dunker:** `rf = rim_xy + 1.0·û(p_dk − rim_xy)`.

| State | Action | Exit |
|---|---|---|
| dispatch (initial) | none | `role.eq(k)` → off / static / jumper / onball / chaser / cutoff / rim |
| static | none | none |
| jumper | none | jump trigger → jump |
| onball | arrive law to `home` | seen → read |
| chaser | intercept law: aim `p_dk + v_dk·(d/speed)`, seek at `speed` | seen → read |
| cutoff | arrive to `p_dk + 0.55·(rf − p_dk)` | seen → read; dunk-contest → jump |
| rim | arrive to `rim_xy + 1.1·û(p_dk − rim_xy) + 0.9·side·perp(û)` (beside the dunker's line to the rim, on the side he is already on; standing in the line would block every take-off at 1.0–1.6 m) | seen → read; dunk-contest → jump |
| read | arrive to `aim_ball` | jump trigger or dunk-contest → jump |
| jump | on_enter: impulse (horizontal velocity kept), `set_var(jump_ready, 0)` | → air |
| air | none | jz < 0.003 and time in state > 0.1 → landed |
| landed | arrive to `rf` at 0.6·speed | none |

- **Arrive law:** `s_des = min(speed, 0.8·sqrt(2·7·max(d − 0.05, 0)))`, then the move law with `max_accel` 7.

### A8 Events and outcome (`rules.py`)
- Every event is a latch, `E = latch(released & cond, time < −1)`, with time `t_E = time.sample_on(E)`.
- `seg_dist(b, axis_xy, z_lo, z_hi) = sqrt(|b_xy − axis_xy|² + max(0, z_lo − b_z, b_z − z_hi)²)`.

| Event | Condition |
|---|---|
| catch | `seg_dist(ball, p_dk + 0.20·û(v_dk,flat), jz_dk + 0.55H, jz_dk + 1.33H) < 0.35`, and visible (`dot(û v_dk,flat, û(ball − p_dk)_flat) ≥ −0.5`), and \|v_ball − v_dk\| ≤ 15, and not `int_any` |
| int_i | `present_i & seg_dist(ball, p_i,xy, 0, jz_i + R_s,i) < 0.37` |
| floor / board / rim / out | z < 0.1393 / inside backboard box grown by r / ring distance < 0.1323 / more than 0.5 m outside the court bounds |
| blocked | not catch, and some present i with jz_i < 0.05 and \|p_i − p_dk\|_xy < 0.55 |
| dunkable | sampled at catch. **Air** (jz_dk ≥ 0.15): ball z ≥ 2.70, \|p_dk − rim\|_xy ≤ 1.4, vz_dk ≥ −1.0, and no airborne defender within 0.9 m whose jz + R_s ≥ ball z − 0.1. **Ground:** dunker in run_t, \|p_dk − T\| ≤ 2.5, no defender within 1.5 m |

- **Resolution** (env, Python, read every 5 ticks; events are latched so nothing is missed): the earliest t_E among
  {catch, int_*, rim, board, floor, out} wins. Tie order: int > catch > rim > board > floor > out.
  - catch → DUNK if dunkable, else CATCH_NO_DUNK
  - int → INTERCEPTED
  - rim / board / floor / out → MISS
  - blocked before release, or before the deciding event → BLOCKED
  - no throw before the dunker reaches land or blocked, or t ≥ 5.0 s → NO_PASS
  - no event within 3.0 s of release → MISS
- **Rewards:**

| Outcome | Reward |
|---|---|
| DUNK | +1.0 |
| CATCH_NO_DUNK | +0.3 |
| INTERCEPTED | −0.5 |
| BLOCKED | −0.5 |
| MISS | −0.3 − 0.2·clip(d_min/1.5, 0, 1), where d_min is the `running_min` of the catch distance after release |
| NO_PASS | −1.0 |

  Outcome codes: 1–6 in this order (0 = none).

### A9 Agent interface (`env.py`, `obs.py`)
- **Decisions** every 10 ticks (0.1 s) until a throw. After a throw the same `step()` runs the wind-up, release and
  flight to resolution, and returns `terminated=True`.
- **Action** `a ∈ [−1, 1]^4`:
  - `a0 ≥ 0` means throw.
  - `ψ = ψ_ref + (π/4)·a1`, with ψ_ref = the bearing from hold to T.
  - `H_ap = 3.40 + 1.55·(a2 + 1)`, so [3.40, 6.50] m.
  - `R = d_ref·(1 + 0.6·a3)`, where d_ref = \|T − hold\|_xy.
  - Inverse (drag-free): `vz = sqrt(2g(H_ap − z0))`, `T_f = vz/g + sqrt(2(H_ap − 3.048)/g)`, `v_h = R/T_f`. If
    \|v0\| > 14, scale v_h down and set `info["clamped"]=True`.
- **Observation** (109 float32, fixed scaling, clipped to ±5; court frame):

| Indices | Content |
|---|---|
| 0–1 | t/3, n_def/5 |
| 2–5 | hold x/10, y/10, z/3, H_p/2.2 |
| 6–28 | dunker: H/2.2, L, R_s/3, x/10, y/10, jz, vx/8, vy/8, vz/5, state one-hot(5: wait, run_c, run_t, jump/air, land/blocked), has_corner, Cx/10, Cy/10, Tx/10, Ty/10, vmax/8, v_gather/8, d_T/10, blocked |
| 29–108 | 5 defender slots × 16: present, x/10, y/10, vx/8, vy/8, jz, vz/5, H/2.2, R_s/3, L, role one-hot(6). Sorted by distance to T; empty slots all zero |

- **Spawn-safe factory** `make_env(rank, base_seed)`. Episode seed is `base_seed + rank·10⁹ + episode_index`. Stage mix
  is set via `venv.call("set_stage_mix", mix)`. Use `AsyncVectorEnv(..., autoreset_mode=SAME_STEP)`.

### A10 Curriculum and gates (`curriculum.py`)
- **Eval sets:** 1024 fixed scenarios per stage (seeds `9_000_000 + 10_000·stage + i`).
- **Gate:** deterministic policy (throw iff p ≥ 0.5, mean parameters), checked every 250k decisions. Two consecutive
  passes promote to the next stage.
- **Mixing:** 75% current stage, 25% uniform over earlier stages.
- **Oracle:** "oracle%" is the oracle success rate on the same set (cached at `runs/lobpass/oracle_<stage>.json`).

| Stage | Gate (success = DUNK rate) | Budget cap (decisions) |
|---|---|---|
| S0 | ≥ 0.95 | 3M |
| S1 | ≥ 0.93 | 4M |
| S2 | ≥ 0.85 × oracle% | 6M |
| S3a | ≥ 0.85 × oracle% | 6M |
| S3b | ≥ 0.80 × oracle% | 8M |
| S3c | ≥ 0.75 × oracle% | 15M |

- **Oracle** (`oracle.py`):
  1. Record the no-throw episode.
  2. For release ticks every 10 ticks, target catch-capsule centers at 0.02 s steps inside both catch windows.
  3. Solve v0 with the drag shooting method (`ballistics.py`, 3 Newton iterations), reject \|v0\| > 14, and rank by
     clearance from defender reach envelopes.
  4. Closed-loop simulate the top 16 in LoBS and take the first DUNK.

  Also reports the share of oracle solutions representable in the action box (target ≥ 97%).

### A11 PPO (`ppo.py`)
- **Networks:** actor and critic are separate MLPs, 109 → 256 → 256 (tanh).
- **Heads:** throw = Bernoulli logit (bias init −2.0); params = Gaussian of 3, state-independent `log_std` init −0.7,
  last layer gain 0.01. The log-prob includes the Gaussian term only when throw = 1.
- **Hyperparameters:**
  - `n_envs` = cpu_count − 2 (min 4), `n_steps` 512
  - `gamma` 0.99, `gae_lambda` 0.95
  - `lr` 3e-4, linear decay within each stage
  - 6 epochs, 4 minibatches
  - `clip` 0.2, `ent_coef` 0.005, `vf_coef` 0.5, `max_grad_norm` 0.5, `target_kl` 0.03 (early stop)
- **Checkpoint:** state dict + config + stage + decision count + `OBS_VERSION` hash.
- **Log:** JSONL per update with stage, decisions, episodes, rolling-2000 outcome shares, reward, entropy, KL,
  clipfrac, value loss, ticks/s.

---

# Part B — Animation and presentation standard (frozen)

### B1 Pipeline and data contracts
- **`record.py`** (show mode, `record=True`, 100 Hz) writes:
  - `episode.npz`: the LoBS `Trajectory`.
  - `episode.json` (`format: "lobpass-episode", version: 1`): seed, stage, checkpoint, scenario (all variables),
    actors (`id, team, role, H, L, R_s, slot`), plan (start, C, T), decisions (`tick, p_throw, a, value`), events
    (`tick, type, actor, data`; types `windup, release, takeoff, land, catch, intercept, deflect, slam, rim_touch, board_touch,
    floor_touch, block`) and outcome.
- **Show mode adds (training never runs these):**
  - Dunk finish: after catch the ball is gripped to the dunker's hands point, steered over the rim front, released by
    a `slam` force law (`rising` pulse, 4 m/s down) when within 0.12 m above the rim center. It falls through the 16
    rim capsules physically.
  - Intercept deflection: the env applies Δv = reflect about the hand normal, restitution 0.5, + 2 m/s up.
- **`anim/bake.py`** → `anim.json` (`format: "lobpass-anim", version: 1`):
  - 60 fps, with characters (rig sizes; per frame pelvis world position + 21 local joint quats as base64 float32
    buffers), ball (pos, quat), camera (pos, target, fov), time-remap curve, overlays and events.
  - The player only renders; all motion logic lives in Python.

### B2 Rig and proportions (fractions of H; the root is the pelvis/hip center)
- **Joints (21):** pelvis, spine, chest, neck, head, clavicle L/R, upper_arm L/R, forearm L/R, hand L/R, thigh L/R,
  shin L/R, foot L/R, toe L/R.
- **Heights:**
  - hip center 0.530; hip lateral ±0.095
  - spine 0.150, chest 0.140 (neck base 0.820), neck 0.050 (head joint 0.870), head height 0.130
  - shoulder joints ±0.115 at 0.815; clavicle elevation up to 0.085 when the arm is overhead
- **Arm:** upper 0.178, forearm 0.148, hand (wrist to fingertip) 0.104. Overhead fingertip = 1.330 = R_s exactly;
  wingspan 1.09·H.
- **Leg:** thigh 0.245, shin 0.246, ankle height 0.039, foot length 0.152 (heel 0.035 behind ankle), toe joint 0.110
  ahead of the heel.
- **Meshes (stylized mannequin; no faces):** rounded boxes for pelvis (0.19×0.11×0.09) and chest (0.24×0.13×0.17);
  head ellipsoid r 0.062 (z ×1.15); capsules for neck r 0.03, upper arm 0.026, forearm 0.022, thigh 0.045→0.038,
  shin 0.032; hand box 0.05×0.025×0.10; shoe rounded box 0.152×0.055×0.04.
- **Joint limits** (also QA): knee 0–150°, elbow 0–155°, hip flexion −20…120°, shoulder elevation 0…180°, neck yaw
  ±70°, pitch ±40°.

### B3 Locomotion (zero foot skate by construction)
- **Root path** = rig xy from LoBS (Hermite-resampled with recorded velocity). The jump comes from the jz joint.
- **Facing:** velocity heading when speed > 0.5 m/s, else the look target. Critically damped spring, ω 12 rad/s, rate
  ≤ 10 rad/s. Defenders face the ball (pre-release: their man/ball holder) and use a **shuffle gait** when
  \|angle(v, facing)\| > 60° (no foot crossing; step 0.40·H).
- **Footstep planner:**
  - Step length `(H/1.9)·(0.45 + 0.24·v)` m, clamped to [0.35H, 1.15H]; cadence = v/step.
  - Duty factor: 0.60 below 2 m/s, 0.38 above 4.5 m/s, linear between.
  - Plants at the root position at plant time ±0.09·H lateral.
  - The stance foot is **fixed in world space** from plant to lift.
  - Swing is a cubic Bézier with apex clearance 0.04H (walk) to 0.10H (sprint), heel strike pitch +15°, toe-off −25°.
- **Pelvis:**
  - Height bob −0.03H at mid-stance, +0.02H in flight.
  - Lateral sway 0.01H toward the stance foot.
  - Yaw counter-rotation ±8° at sprint.
  - Forward lean `0.6·atan(a_fwd/g) + 6°·v/7`.
  - Turn lean `atan(v·ω/g)`, clamped to 25°.
- **Legs:** two-bone IK with a forward knee pole. If a foot target is out of reach, lower the pelvis (never let the
  foot float).
- **Arms while running:** opposite-arm swing ±35°, elbow 90°. Action layers override them.

### B4 Action layers (blend in/out with smoothstep over 0.12 s unless stated)

| Layer | Timing (relative to sim ticks) | Pose rules |
|---|---|---|
| Passer idle | until windup | triple-threat; ball at chest; knees 20°; head tracks dunker |
| Pass | windup = the 15 sim ticks before release; release frame exact; follow-through 0.35 s | ball path chest → behind head (+0.05H, −0.15 m) → hold point **exactly at the release tick**; arms extend along ψ; wrist snap |
| Jump (all) | anticipation crouch 0.15 s before takeoff tick; airborne from jz; landing absorb 0.12 s + recover 0.25 s | crouch pelvis −0.08H, knees 60°, arms back; tuck knees 40°; land pelvis −0.10H |
| Catch | reach starts 0.30 s before the catch tick | two-hand IK to the ball's (known) catch-tick position; hands-mid = ball center at the catch frame, palms 0.24 m apart |
| Dunk | ball carried by hands IK on the sim ball until `slam`; then one-hand rim hang 0.25 s if still airborne | strong arm over the rim; legs split slightly; head up |
| Defender stance | whenever not running fast | feet 0.30H apart, knees 35°, hands up (ONBALL/STATIC overhead; others at shoulder) |
| Contest / intercept | arm raise 0.25 s before the jump apex or intercept tick | one-arm IK to the ball position at that tick; hand within 5 cm of the ball surface at intercept |
| Look-at | always | head/neck toward the ball, clamped, ω 10 rad/s |
| Idle breathing | always | chest scale ±1% at 0.25 Hz |

- **Ball visual:**
  - While held, ball = hands-mid. At the release tick the visual ball must equal the sim ball (≤ 2 cm frame-to-frame jump).
  - After release, ball = sim ball exactly.
  - Spin: backspin 2.5 rev/s about `cross(up, v̂)` in flight; sim angular velocity after the first contact.

### B5 Hoop, net, court look
- **Net:** runtime Verlet in the player at a fixed 120 Hz substep (deterministic):
  - 12 strands × 6 knots, length 0.42 m, top ring r 0.2286, bottom ring r 0.14
  - 4 constraint iterations, damping 0.985
  - ball-sphere collision (r + 0.01)
  - seeded; no `Math.random`
- **Rim flex** on slam: the front dips 3°, damped spring ω 25 rad/s, ζ 0.25 (visual only).
- **Court:**
  - Procedural maple texture (canvas: 0.06 m planks, two-tone grain, seeded), clear-coat roughness 0.35.
  - Paint #9E2A2B, lines white 0.0508 m (drawn from `court.py` polylines), apron #2B2B2B beyond the bounds.
  - Backboard is glass (transmission 0.9, white border and target square); rim is orange #D9531E metallic 0.6;
    stanchion has dark padding.
- **Teams:** offense jersey #1F4FD1 / shorts #163A9E; defense #F2F2F2 / #C9C9C9; skin neutral #C8A27E; shoes white
  with team-colored sole.

### B6 Camera director (`anim/camera.py`, baked)
- **Shots:**
  - **Broadcast:** sideline at (5.5, −15.5, 5.2), FOV 38°, look-at = spring-follow (ω 2.5) of the midpoint of ball,
    dunker and rim.
  - **Release:** over the passer's shoulder, from 0.4 s before release to 0.3 s after.
  - **Flight:** track the ball with 0.25 s lead.
  - **Baseline low:** (−3.0, ±1.5, 1.1), FOV 50°, for slow motion.
- **Rules:**
  - No cut within 0.25 s of release or catch.
  - Obey the 180° line (passer→rim) except the replay intro.
  - Pan rate ≤ 60°/s.
  - Hard cuts only (broadcast style).
- **Time remap:** 1.0× normally; replay segment at 0.25× from catch − 0.35 s to slam + 0.4 s; ramps are smoothstep over 0.15 s.

### B7 Lighting and render
- Hemisphere light (sky #DDE6FF, ground #3A2E22, intensity 0.6).
- Key directional light (intensity 2.2, from above half court, PCFSoft shadows 2048², bounds = court).
- 4 arena spots (intensity 1.2, cone 35°, no shadows).
- ACES Filmic tone mapping, exposure 1.0, sRGB output, MSAA, pixel ratio 2 when capturing.
- Background: vertical gradient #0E1117→#1B2130 with a soft vignette.

### B8 HUD and overlays (toggleable; on in the reel)
- **Corner card:** stage label, episode seed, and outcome badge (DUNK green / MISS gray / INTERCEPTED red).
- **Decision strip:** a sparkline of p_throw over time, with a marker at release.
- **Release readout:** speed, launch angle, apex height.
- **Path overlays:** the dunker's planned path as a dashed floor line; the predicted drag-free arc as a dotted line
  fading after release.
- **Optional:** defender reach cylinders at 15% opacity.
- **Font:** Inter (Google Fonts, vendored), white on 60% black plates, 8 px radius.

### B9 Video and reel
- 1920×1080, 60 fps, H.264 High, yuv420p, CRF 16, BT.709. No audio.
- Deterministic capture: the player advances by exact frame steps in `?capture=1` mode (no real-time clock).
- **Reel** (60–90 s):
  1. Title (2 s).
  2. The same S3c scenario played by checkpoints at 0% / 10% / 40% / 100% (learning montage).
  3. One highlight per stage S0→S3c with its stage card.
  4. Hero play: 5 defenders, broadcast shot, then a baseline 0.25× replay.
  5. Overlay explainer of the decision moment.
  6. End card.

### B10 Animation QA (`anim/qa.py`; pytest on fixture episodes and every reel episode)

| Check | Limit |
|---|---|
| Planted-foot drift during stance | ≤ 1.0 cm |
| Foot below floor | ≥ −0.5 cm; grounded stance foot z ≤ 1 cm |
| Joint limits (B2) | never exceeded |
| Bone lengths | constant to 0.1 mm |
| Root vs sim at sim ticks | xy ≤ 1 mm; jump ≤ 1 mm |
| Catch frame | hands-mid to ball center ≤ 3 cm; palm gap 0.24 ± 0.02 m |
| Intercept frame | hand to ball surface ≤ 5 cm |
| Release continuity | visual ball jump ≤ 2 cm/frame around release |
| Pose pops | per-joint change ≤ 25°/frame |
| Overhead fingertip | 1.33H ± 2 cm |
| Character overlap (capsule proxies) | ≤ 2 cm except at block/contest events |
| Camera | no cut inside protected windows; pan ≤ 60°/s |

---

