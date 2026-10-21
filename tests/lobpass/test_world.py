import numpy as np
import pytest

import lobs
from demos.lobpass import ballistics as B
from demos.lobpass import constants as C
from demos.lobpass.bodies import jump_cmd_vz
from demos.lobpass.world import build_world, dyn_slots

HOLD_Z = 1.25 * C.PASSER_H_S0


@pytest.fixture(scope="module")
def world():
    return build_world("train")


def make(world):
    sim = world.simulate("mujoco", seed=0, record=False)
    return sim, dyn_slots(sim)


def reset_ball(sim, pos, vel=(0, 0, 0), passer=(5.0, 0.0), **others):
    init = {
        "ball": lobs.Initial(pos=pos, linvel=vel),
        "passer": lobs.Initial(q={"jx": passer[0], "jy": passer[1]}),
    }
    init.update(others)
    sim.reset(seed=0, initial_state=init)


def test_native_core(world):
    sim, _ = make(world)
    assert sim.info["native_core"] is True


def test_check_world_passes(world):
    res = lobs.check_world(world, "mujoco", seconds=2.0)
    assert res.passed, res.summary()


def test_drag_flight_matches_ballistics(world):
    sim, s = make(world)
    p0, v0 = np.array([5.0, 0.0, 2.0]), np.array([9.0, 3.0, 9.0])
    reset_ball(sim, tuple(p0), tuple(v0))
    ps, _ = B.simulate(p0, v0, 170)
    worst = 0.0
    for k in range(170):
        sim.step()
        worst = max(worst, float(np.linalg.norm(sim.state().body_pos[s["ball"]] - ps[k + 1])))
    assert worst < 0.01, worst


def test_throw_impulse_gives_v0(world):
    sim, s = make(world)
    reset_ball(sim, (5.0, 0.0, HOLD_Z))
    for _ in range(20):
        sim.step()  # held: ball stays at the hold point
    assert np.linalg.norm(sim.state().body_pos[s["ball"]] - [5.0, 0.0, HOLD_Z]) < 0.01
    v0 = np.array([4.0, -3.0, 7.5])
    sim.apply_force("ball", C.BALL_MASS * v0 / C.TICK, ticks=1)
    sim.step()
    assert np.linalg.norm(sim.state().body_linvel[s["ball"]] - v0) < 1e-3
    sim.step()
    assert bool(sim["released"])


def jump(sim, name, lift, vh=(0.0, 0.0)):
    vz = jump_cmd_vz(lift)
    v = sim.state().body_linvel[dyn_slots(sim)[name]]
    f = C.PLAYER_MASS * np.array(
        [(vh[0] - v[0]) / C.TICK, (vh[1] - v[1]) / C.TICK, vz / C.TICK + C.G]
    )
    sim.apply_force(f"{name}/body", f, ticks=1)


@pytest.mark.parametrize("lift", [0.5, 0.8, 0.95])
def test_jump_apex_and_airtime(world, lift):
    sim, s = make(world)
    sim.reset(seed=0)
    jump(sim, "dunker", lift)
    zs = []
    for _ in range(120):
        sim.step()
        zs.append(sim.state().body_pos[s["dunker"]][2])
    zs = np.array(zs)
    assert abs(zs.max() - lift) < 0.01, zs.max()
    air = (zs > 0.002).sum() * C.TICK
    assert abs(air - 2 * (2 * lift / C.G) ** 0.5) < 0.02, air


def test_grounded_jz_stays_in_band(world):
    sim, s = make(world)
    sim.reset(seed=0)
    lo, hi = 9.0, -9.0
    for _ in range(300):
        sim.step()
        z = sim.state().body_pos[[s[p] for p in ("passer", "dunker", "d0", "d4")]][:, 2]
        lo, hi = min(lo, z.min()), max(hi, z.max())
    assert C.GROUNDED_JZ[0] <= lo and hi <= C.GROUNDED_JZ[1], (lo, hi)


def test_players_cannot_overlap(world):
    sim, s = make(world)
    sim.reset(seed=0, initial_state={"dunker": lobs.Initial(q={"jx": 6.0, "jy": 0.0}),
                                     "d0": lobs.Initial(q={"jx": 7.0, "jy": 0.0})})  # fmt: skip
    gap = 9.0
    for _ in range(150):
        sim.apply_force("dunker/body", (700, 0, 0))
        sim.apply_force("d0/body", (-700, 0, 0))
        sim.step()
        p = sim.state().body_pos
        gap = min(gap, float(np.linalg.norm(p[s["dunker"]][:2] - p[s["d0"]][:2])))
    assert gap > 2 * C.PLAYER_CAPSULE_R - 0.02, gap


def test_ball_through_rim_centre(world):
    sim, s = make(world)
    reset_ball(sim, (0.0, 0.0, 4.2), (0, 0, -3.0))
    for _ in range(100):
        sim.step()
        if sim.state().body_pos[s["ball"]][2] < 2.5:
            break
    p = sim.state().body_pos[s["ball"]]
    assert p[2] < 2.5 and np.linalg.norm(p[:2]) < 0.05, p


def test_ball_rims_out_on_the_ring(world):
    sim, s = make(world)
    reset_ball(sim, (C.RIM_RING_R, 0.0, 4.2), (0, 0, -3.0))  # straight onto the tube
    for _ in range(60):
        sim.step()
    assert (
        sim.state().body_pos[s["ball"]][2] > 2.9
        or np.linalg.norm(sim.state().body_pos[s["ball"]][:2]) > 0.3
    )


def test_yaml_round_trip_is_identical(world, tmp_path):
    path = tmp_path / "w.yaml"
    world.save(path)
    again = lobs.load_world(path)

    def run(w):
        sim, s = make(w)
        reset_ball(sim, (5.0, 0.0, 2.0), (6.0, 1.0, 5.0))
        for _ in range(100):
            sim.step()
        return sim.state().body_pos.copy()

    assert np.allclose(run(world), run(again), atol=1e-9)


def test_solve_v0_hits_target():
    p0 = np.array([6.0, 2.0, 2.4])
    target = np.array([1.2, -0.5, 3.6])
    v, miss = B.solve_v0(p0, target, 110)
    assert miss < 0.01
    assert np.linalg.norm(B.simulate(p0, v, 110)[0][-1] - target) < 0.01


def test_clamp_speed():
    v, c = B.clamp_speed(np.array([20.0, 0.0, 6.0]))
    assert c and abs(np.linalg.norm(v) - C.MAX_THROW_SPEED) < 1e-9 and v[2] == 6.0
    v2, c2 = B.clamp_speed(np.array([3.0, 0.0, 5.0]))
    assert not c2


def test_batch_solver_matches_lobs_release(world):
    """A throw solved with the batch model reaches its target in the real simulation."""
    hold = np.array([5.0, 0.0, HOLD_Z])
    target = np.array([1.5, 0.8, 3.1])
    n = 100
    v0, miss = B.solve_release_batch(hold, target[None], np.array([n]))
    assert miss[0] < 1e-3
    sim, s = make(world)
    reset_ball(sim, tuple(hold))
    for _ in range(10):
        sim.step()
    sim.apply_force("ball", C.BALL_MASS * v0[0] / C.TICK, ticks=1)
    for _ in range(n):
        sim.step()
    assert np.linalg.norm(sim.state().body_pos[s["ball"]] - target) < 0.01
