# Lob pass: an agent learns to throw an alley-oop through defenders

A LoBS demo. A **passer** (the learning agent) stands still with the ball. A **dunker**, its teammate, runs a fixed
path toward the rim. **Defenders** try to block his path and to pick off the pass. The passer decides *when* to throw
and *which lob* to throw (direction, height, range) so the dunker catches the ball, on the run or in the air, and puts
it in. Everything physical (ball with air drag, players, rim, backboard, floor) is the real LoBS/MuJoCo world; the
dunker and the defenders are declarative LoBS state machines.

```
demos/lobpass/
  world.py      the one LoBS world: court, hoop, ball (drag + hold), seven player rigs, state machines, event measures
  machines.py   dunker and defender state machines          rules.py     catch / intercept / rim / block event measures
  scenario.py   seeded scenario samplers per stage          planner.py   the dunker's path (one corner at most)
  env.py        the Gymnasium environment (decisions every 0.1 s, one throw per play)
  oracle.py     a scripted expert: finds a dunking release for the evaluation sets and for imitation
  bc.py         behaviour cloning from the oracle          ppo.py       PPO with a hybrid (throw / parameters) action
  train.py      curriculum PPO, gates, checkpoints         pipeline.py  the whole curriculum, resumable
  evaluate.py   per-stage metrics and plots                vec.py       batched multi-process vector env
  record.py     a play -> episode.npz / episode.json       anim/        bake: skeleton, gait, IK, ball, camera, QA
  player/       Three.js renderer                          capture.py   deterministic frame capture -> MP4
  reel.py       learning montage, stage highlights, hero play, cards     studio.py   run the passer inside LoBS Studio
  SPEC.md       every number, frozen
```

## Run it

```powershell
pixi install -e lob                                # torch (CPU), playwright, ffmpeg, matplotlib
pixi run -e lob python -m playwright install chromium
pixi run python examples/12_lobpass.py             # one scripted play, prints the outcome

pixi run lob-test                                  # 100+ tests (physics, scenarios, defenders, animation QA, tools)
pixi run -e lob lob-bench                          # throughput of the training world
pixi run -e lob python -m demos.lobpass.pipeline   # train every stage (BC warm-start + PPO, gated)
pixi run -e lob lob-eval --checkpoint runs/lobpass/ft_S3c/ckpt_S3c_final.pt --out runs/lobpass/eval_final
pixi run -e lob python -m demos.lobpass.reel --final runs/lobpass/ft_S3c/ckpt_S3c_final.pt --out runs/lobpass/footage/reel
pixi run -e lob python -m demos.lobpass.studio --checkpoint runs/lobpass/ft_S3c/ckpt_S3c_final.pt --stage S3c --seed 3
```

## The task

* **Court and hoop** are NBA size (3.048 m rim, backboard, 16-capsule rim ring), court frame with the origin under the rim.
* **The ball** is held by a spring force at the passer's hold point; a throw is a one-tick release impulse. Flight has
  quadratic drag (C = 0.0135 kg/m), so range and apex differ from the drag-free formula the action uses.
* **Players** are fixed-root actors on x/y/z slide joints, driven by force laws with acceleration limits. Per-episode heights
  live in behavior variables; the collision proxy is a 2 m capsule.
* **The passer's action** (4 numbers in [-1, 1]): throw now?, bearing offset, apex height (3.4-6.5 m) and range. One decision
  every 0.1 s; a 0.15 s wind-up is locked in once it throws.
* **Outcomes and reward**: DUNK +1, caught but no dunk +0.3, intercepted -0.5, blocked -0.5, miss -0.3 (shaped by how close),
  never passed -1.
* **Curriculum**: S0 fixed heights, no defenders; S1 random heights; S2 stationary defenders; S3a defenders that read the
  ball and jump; S3b defenders that chase and cut off; S3c five fast defenders. Each stage has a gate on a fixed set of
  1024 scenarios, relative to what the oracle achieves where defenders can make a dunk impossible.

## How it is trained (and what we found)

Pure PPO from scratch learns, but slowly: **67 %** deterministic DUNK on S0 after 1.25 M decisions (24 min), then it
flattens. The success region is tiny and a per-step Bernoulli throw decision makes sampled plays throw at random times. So every
stage starts from **behaviour cloning of the scripted oracle** and is then fine-tuned with PPO, keeping the best deterministic
evaluation. PPO fine-tuning never beat the clone here (it lowered every stage: S2 65 -> 46 %, S3a 66 -> 48 %, S3b 21 -> 13 %),
so the policies below are the imitation policies and the PPO phase is an honest negative result.

Deterministic policy on the fixed 1024-scenario evaluation set of each stage (`eval_stagewise`):

| Stage | Dunk | Oracle ceiling | Gate | Notes |
|---|---:|---:|---:|---|
| S0 fixed heights, open court | **95.9 %** | 100 % | 95 % | passed (imitation) |
| S1 random heights | **94.9 %** | 100 % | 93 % | passed (imitation) |
| S2 stationary defenders | 65.3 % | 99.5 % | 84.6 % | missed; interceptions 22 % |
| S3a defenders that jump | 66.1 % | 99.6 % | 84.7 % | missed |
| S3b chase and cut off | 20.6 % | 29.6 % | 23.7 % | missed by 3 points; 29 % never pass |
| S3c five fast defenders | 10.0 % | 14.3 % | 10.7 % | missed by 0.7 points; 70 % of the oracle |

Where the oracle itself only dunks 14-30 % of plays (a defender can simply wall the runner in), the learned passer reaches
70 % of what is possible. What did not work, and why it is left in: S2-S3a fall well short of their gates (interceptions:
the clone's small timing and height errors land the ball inside a defender's reach where the oracle's did not); each stage's
policy forgets the earlier stages (the S3c policy dunks 27 % of S0); and PPO fine-tuning at these sample sizes (about 300
informative plays per update) degrades rather than refines a good clone.

Throughput: 28 k simulated ticks/s aggregate (8.2 k per process) on a power-limited laptop CPU, with the native LoBS tick
core. Footage lives in `runs/lobpass/footage/reel/` (not committed): `lobpass_reel.mp4` (1920x1080, 60 fps, 101 s),
`training_footage.mp4` and `post_training_footage.mp4`.

## Animation

Physics is the source of truth, but it is a handful of capsules. For the footage a Python baker turns each recorded play
into a full-body animation, and a Three.js page renders it:

* **Skeleton**: 21 joints with SPEC B2 proportions (overhead fingertip = 1.33 H exactly).
* **Locomotion with no foot skate by construction**: a planted foot is a fixed world point from touch-down to lift-off; each
  plant is placed under where the root will be at mid-stance; hips are kept within reach by lowering the pelvis, never by
  stretching a leg.
* **IK for catches, blocks and the ball in hand**, so the palms are on the ball at the catch frame (measured: 0.000 cm).
* **Ball**: in the hands before the release (up and over the head, exactly the sim's hold point at the release tick), the sim's
  physical flight afterwards, a cushioned catch, a carried finish, a dunk through a deterministic cloth net.
* **Director camera**: broadcast, over-the-shoulder at the release, slow-motion for the decisive moment, hard cuts only and
  never within 0.25 s of the release or the catch; the pan rate is limited.
* **QA** (`anim/qa.py`, `tests/lobpass/test_anim.py`): foot drift, floor contact, joint limits, bone lengths, root fidelity,
  catch and intercept contact, release continuity, pose pops, overlap and camera rules are measured on five plays (alley-oop,
  ground-catch dunk, interception, miss, block) in the test suite.
