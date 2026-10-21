import dataclasses

import numpy as np
import pytest

from demos.lobpass import ballistics as B
from demos.lobpass import constants as C
from demos.lobpass import oracle, planner
from demos.lobpass.env import LobPassEnv
from demos.lobpass.machines import DEFENDER_STATES
from demos.lobpass.scenario import (
    ROLE_CHASER, ROLE_CUTOFF, ROLE_JUMPER, ROLE_ONBALL, ROLE_RIM, ROLE_STATIC, DefenderSpec, sample_scenario,
)  # fmt: skip
from demos.lobpass.world import build_world, dyn_slots

NO_THROW = np.array([-1.0, 0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def env():
    return LobPassEnv({"S0": 1.0})


def spec(slot, role, xy, speed=5.0, react=0.2, H=2.0, L=0.7):
    return DefenderSpec(slot, role, (float(xy[0]), float(xy[1])), H, L, speed, react)


def with_defenders(base, defs):
    return dataclasses.replace(base, defenders=list(defs))


def run_free(env, scn, seconds, probe):
    """Run without throwing; ``probe(env)`` is called every tick and its results collected."""
    env.reset(options={"scenario": scn})
    out = []
    for _ in range(int(seconds / C.TICK)):
        env.sim.step()
        out.append(probe(env))
    return out


def dstate(env, i):
    return DEFENDER_STATES[int(env.sim.reg and env.sim._beh.by_name[f"d{i}"].state_idx)]


def pos(env, name):
    return env.sim.state().body_pos[env.slots[name]].copy()


def test_world_stays_native_with_five_defender_machines():
    sim = build_world("train").simulate("mujoco", record=False)
    assert sim.info["native_core"] is True
    assert len(dyn_slots(sim)) == 8  # ball, passer, dunker, five defenders


def test_planner_keeps_clear_of_stationary_defenders():
    good = 0
    for seed in range(120):
        s = sample_scenario("S2", seed)
        pts = [np.array(s.dunker_start)] + ([np.array(s.corner)] if s.corner else []) + [np.array(s.takeoff)]
        c = planner.clearance(pts, np.array([d.start for d in s.defenders]))
        good += float(c.min()) >= 1.0
    assert good >= 0.85 * 120


def test_static_defender_stays_put_and_intercepts_low_but_not_high_passes(env):
    base = sample_scenario("S1", 4)
    rec = oracle.record(env, base)
    env.reset(options={"scenario": base})
    hold = env.hold
    m = int(np.argmax(rec.dk_state >= 2) + 20)
    # flat pass, defender planted on its line: intercepted
    target = np.array([*rec.dk_pos[m][:2], 2.2])
    v0, _ = B.solve_release_batch(hold, target[None], np.array([60]))
    mid = B.batch_positions(hold, v0, np.array([60]), np.array([[30]]))[0, 1]
    low = with_defenders(base, [spec(0, ROLE_STATIC, mid[:2], speed=0.0, react=0.0, H=2.2)])
    env.reset(options={"scenario": low})
    for _ in range(8):
        env.step(NO_THROW)
    assert env.throw(v0[0])[4]["outcome"] == "INTERCEPTED"
    # the oracle's lob passes well above a static defender standing under its midpoint: not intercepted
    sol = oracle.solve(env, base)
    path = B.batch_positions(hold, sol["v0"][None], np.array([100]), np.array([[50]]))[0, 1]
    assert path[2] > 2.2 * C.REACH_FRAC + 0.2
    high = with_defenders(base, [spec(0, ROLE_STATIC, path[:2], speed=0.0, react=0.0, H=2.2)])
    env.reset(options={"scenario": high})
    start = pos(env, "d0")[:2].copy()
    for _ in range((sol["release"] - C.WINDUP_TICKS) // C.DECISION_TICKS):
        env.step(NO_THROW)
    assert np.linalg.norm(pos(env, "d0")[:2] - start) < 0.05  # it never moved
    assert env.step(sol["action"])[4]["outcome"] != "INTERCEPTED"


def test_jumper_apex_lines_up_with_the_ball_crossing(env):
    base = sample_scenario("S1", 4)
    sol = oracle.solve(env, base)
    env.reset(options={"scenario": base})
    hold = env.hold
    n = 100
    ticks = np.arange(1, n + 1)
    path = np.array([B.batch_positions(hold, sol["v0"][None], np.array([k]))[0] for k in ticks])
    H, L = 2.0, 0.7
    z_reach = C.REACH_FRAC * H + L - C.BALL_READ_MARGIN
    k = int(np.argmax((path[:, 2] < z_reach) & (np.arange(n) > n // 2)))  # descending crossing of the reach height
    cross_tick = sol["release"] + int(ticks[k])
    jumper = with_defenders(base, [spec(0, ROLE_JUMPER, path[k][:2], speed=0.0, react=0.3, H=H, L=L)])
    env.reset(options={"scenario": jumper})
    for _ in range((sol["release"] - C.WINDUP_TICKS) // C.DECISION_TICKS):
        env.step(NO_THROW)
    zs = []
    env.sim.step  # noqa: B018  (keeps the stepping loop below explicit)
    env.hold = env.hold
    # play the throw manually so the defender trace can be recorded tick by tick
    for _ in range(C.WINDUP_TICKS):
        env.sim.step()
        zs.append(pos(env, "d0")[2])
    vb = env._ball_vel()
    env.sim.apply_force("ball", C.BALL_MASS * (sol["v0"] - vb) / C.TICK, ticks=1)
    for _ in range(n + 40):
        env.sim.step()
        zs.append(pos(env, "d0")[2])
    apex_tick = int(np.argmax(zs)) + (sol["release"] - C.WINDUP_TICKS)
    assert max(zs) > 0.5 * L  # it jumped
    assert abs(apex_tick - cross_tick) * C.TICK <= 0.25, (apex_tick, cross_tick)


def test_chaser_closes_on_a_slower_dunker(env):
    base = sample_scenario("S1", 6)
    d0 = np.array(base.dunker_start) + np.array([2.2, 0.0])
    scn = with_defenders(base, [spec(0, ROLE_CHASER, d0, speed=6.0)])
    scn = dataclasses.replace(scn, vmax=5.0, v_corner=3.5, start_delay=0.0)
    dist = run_free(env, scn, 3.0, lambda e: np.linalg.norm(pos(e, "d0")[:2] - pos(e, "dunker")[:2]))
    assert min(dist) < 0.9 or dist[-1] < dist[0] - 1.0


def test_cutoff_stays_between_dunker_and_rim(env):
    base = sample_scenario("S1", 6)
    start = np.array(base.dunker_start)
    seg = np.array(base.takeoff) - start
    side = start + 0.6 * seg + 2.5 * np.array([-seg[1], seg[0]]) / np.linalg.norm(seg)
    scn = with_defenders(base, [spec(0, ROLE_CUTOFF, side, speed=5.5)])
    scn = dataclasses.replace(scn, start_delay=0.0)

    def off_line(e):
        d, r = pos(e, "dunker")[:2], np.zeros(2)
        p = pos(e, "d0")[:2]
        t = np.clip(np.dot(p - d, r - d) / max(np.dot(r - d, r - d), 1e-9), 0, 1)
        return np.linalg.norm(p - (d + t * (r - d)))

    off = run_free(env, scn, 3.5, off_line)
    assert min(off[100:]) < 1.5


def test_rim_protector_contests_an_alley_oop(env):
    base = sample_scenario("S1", 3)
    T = np.array(base.takeoff)
    ang = np.arctan2(T[1], T[0]) + np.radians(55.0)  # starts beside the take-off line
    scn = with_defenders(base, [spec(0, ROLE_RIM, 1.1 * np.array([np.cos(ang), np.sin(ang)]), speed=5.0, react=0.2)])
    scn = dataclasses.replace(scn, start_delay=0.0, corner=None)
    states = run_free(env, scn, 9.0, lambda e: (int(e.sim["dunker_state"]), env.sim._beh.by_name["d0"].state_idx))
    dk_jump = next(k for k, (d, _) in enumerate(states) if d == 3)
    rim_jump = next((k for k, (_, r) in enumerate(states) if DEFENDER_STATES[r] == "jump"), None)
    assert rim_jump is not None and 0 <= rim_jump - dk_jump <= 20, (dk_jump, rim_jump)


def test_reaction_delay_is_honored(env):
    base = sample_scenario("S1", 4)
    sol = oracle.solve(env, base)
    react = 0.3
    scn = with_defenders(base, [spec(0, ROLE_ONBALL, np.array(base.passer_xy) * 0.8, speed=3.0, react=react)])
    env.reset(options={"scenario": scn})
    for _ in range((sol["release"] - C.WINDUP_TICKS) // C.DECISION_TICKS):
        env.step(NO_THROW)
    for _ in range(C.WINDUP_TICKS):
        env.sim.step()
    env.sim.apply_force("ball", C.BALL_MASS * (sol["v0"] - env._ball_vel()) / C.TICK, ticks=1)
    t_rel = env.sim.time
    left = None
    for _ in range(80):
        env.sim.step()
        if left is None and DEFENDER_STATES[env.sim._beh.by_name["d0"].state_idx] != "onball":
            left = env.sim.time
    assert left is not None
    # reaction time, plus up to one latency and one decision period of machine delay
    assert react - 0.01 <= left - t_rel <= react + C.DEF_LATENCY + C.DEF_PERIOD + 0.03, left - t_rel


def test_defenders_keep_their_distance_from_each_other(env):
    worst = 9.0
    for seed in range(6):
        s = sample_scenario("S3c", seed)
        if s.n_def < 2:
            continue
        env.reset(options={"scenario": s})
        names = [f"d{d.slot}" for d in s.defenders] + ["dunker"]
        for _ in range(500):
            env.sim.step()
            p = np.array([pos(env, n)[:2] for n in names])
            dm = np.linalg.norm(p[:, None] - p[None], axis=-1) + np.eye(len(p)) * 99
            worst = min(worst, float(dm.min()))
    assert worst > 0.44, worst


def test_body_block_stops_the_dunker_and_yields_blocked(env):
    base = sample_scenario("S1", 4)
    base = dataclasses.replace(base, corner=None, start_delay=0.0)
    start, T = np.array(base.dunker_start), np.array(base.takeoff)
    blocker = spec(0, ROLE_STATIC, start + 0.6 * (T - start), speed=0.0, react=0.0)
    scn = with_defenders(base, [blocker])
    env.reset(options={"scenario": scn})
    for _ in range(80):
        _, _, term, _, info = env.step(NO_THROW)
        if term:
            break
        if int(env.sim["dunker_state"]) == 6:
            break
    assert int(env.sim["dunker_state"]) == 6
    assert bool(env.sim["ev_blocked"])
    _, r, term, _, info = env.throw(np.array([2.0, 0.0, 6.0]))
    assert term and info["outcome"] == "BLOCKED" and r == -0.5
