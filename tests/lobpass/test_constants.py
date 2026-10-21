import math

from demos.lobpass import constants as C


def test_drag_coefficient():
    assert abs(C.DRAG_C - 0.013459) < 1e-6


def test_tick_is_ten_ms():
    assert abs(C.TICK - 0.01) < 1e-12


def test_overhead_fingertip_matches_reach():
    assert C.OVERHEAD_FINGERTIP == C.REACH_FRAC
    # overhead chain: shoulder + clavicle lift + arm segments = 1.33 H
    chain = C.SHOULDER_Z + C.CLAVICLE_LIFT + C.ARM_UPPER + C.ARM_FORE + C.ARM_HAND
    assert abs(chain - C.OVERHEAD_FINGERTIP) < 1e-9


def test_leg_chain_reaches_hip():
    assert abs(C.ANKLE_Z + C.LEG_SHIN + C.LEG_THIGH - C.HIP_Z) < 0.02


def test_rim_geometry():
    assert abs(C.RIM_TUBE_Z + C.RIM_TUBE_R - C.RIM_TOP_Z) < 1e-9
    assert abs(C.RIM_RING_R - C.RIM_INNER_R - C.RIM_TUBE_R) < 1e-9
    assert C.BOARD_FRONT_X == C.BOARD_CENTER[0] + C.BOARD_SIZE[0] / 2


def test_jump_helpers():
    assert abs(C.jump_vz(0.8) ** 2 / (2 * C.G) - 0.8) < 1e-12
    assert math.isclose(C.reach(1.98), 1.33 * 1.98)


def test_obs_layout():
    assert 29 + C.N_DEF_SLOTS * C.DEF_OBS == C.OBS_DIM
