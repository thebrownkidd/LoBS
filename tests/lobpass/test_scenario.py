import math

import numpy as np
import pytest
from scipy import stats

from demos.lobpass import constants as C
from demos.lobpass import court, planner
from demos.lobpass.scenario import ROLE_STATIC, sample_scenario


@pytest.mark.parametrize("stage", C.STAGES)
def test_deterministic_and_in_bounds(stage):
    n = 400
    for seed in range(n):
        a, b = sample_scenario(stage, seed), sample_scenario(stage, seed)
        assert a == b
        pts = (
            [a.passer_xy, a.dunker_start, a.takeoff]
            + ([a.corner] if a.corner else [])
            + [d.start for d in a.defenders]
        )
        for x, y in pts:
            assert court.in_bounds(x, y, C.BOUNDS_MARGIN - 1e-9), (stage, seed, (x, y))
        assert (
            np.linalg.norm(np.subtract(a.passer_xy, a.dunker_start))
            >= C.DUNKER_MIN_DIST_PASSER - 1e-9
        )
        assert C.TAKEOFF_R[0] - 1e-9 <= np.linalg.norm(a.takeoff) <= C.TAKEOFF_R[
            1
        ] + 1e-9 or stage not in ("S0", "S1")
        assert a.dunker_L >= C.DUNK_REACH_Z - a.dunker_R_s - 1e-9 or stage == "S0"


def test_stage_defender_counts_and_roles():
    for seed in range(300):
        assert sample_scenario("S0", seed).n_def == 0 and sample_scenario("S1", seed).n_def == 0
        s2 = sample_scenario("S2", seed)
        assert 1 <= s2.n_def <= 4 and all(d.role == ROLE_STATIC for d in s2.defenders)
        assert 1 <= sample_scenario("S3b", seed).n_def <= 3
        assert 2 <= sample_scenario("S3c", seed).n_def <= 5


def test_defender_spacing_constraints():
    for seed in range(300):
        s = sample_scenario("S3c", seed)
        ds = [np.array(d.start) for d in s.defenders]
        for i, p in enumerate(ds):
            assert np.linalg.norm(p - s.dunker_start) >= C.DEF_MIN_DUNKER - 1e-9
            assert np.linalg.norm(p - s.passer_xy) >= C.DEF_MIN_PASSER - 1e-9
            for q in ds[:i]:
                assert np.linalg.norm(p - q) >= C.DEF_MIN_SEP - 1e-9


def test_distributions_match_spec():
    ss = [sample_scenario("S1", s) for s in range(1500)]
    vmax = np.array([s.vmax for s in ss])
    assert stats.kstest((vmax - C.VMAX[0]) / (C.VMAX[1] - C.VMAX[0]), "uniform").pvalue > 0.01
    h = np.array([s.passer_H for s in ss])
    assert (
        stats.kstest(
            (h - C.PASSER_H_RANGE[0]) / (C.PASSER_H_RANGE[1] - C.PASSER_H_RANGE[0]), "uniform"
        ).pvalue
        > 0.01
    )
    corner_share = np.mean([s.corner is not None for s in ss])
    assert 0.35 < corner_share < 0.55  # 50 % corner style, minus failed corner samples


def test_corner_turn_within_spec():
    seen = 0
    for seed in range(500):
        s = sample_scenario("S1", seed)
        if s.corner is None:
            continue
        seen += 1
        a, c, b = map(np.array, (s.dunker_start, s.corner, s.takeoff))
        cosang = np.dot(c - a, b - c) / np.linalg.norm(c - a) / np.linalg.norm(b - c)
        assert (
            C.CORNER_TURN_DEG[0] - 1e-6
            <= math.degrees(math.acos(cosang))
            <= C.CORNER_TURN_DEG[1] + 1e-6
        )
    assert seen > 100


def test_planner_deterministic_valid_and_avoids_defenders():
    start = np.array([8.0, -3.0])
    t1, c1, j1 = planner.plan(start, np.zeros((0, 2)), 6.0)
    t2, c2, j2 = planner.plan(start, np.zeros((0, 2)), 6.0)
    assert np.array_equal(t1, t2) and j1 == j2 and (c1 is None) == (c2 is None)
    assert c1 is None  # nothing in the way: straight line wins
    # a defender planted on the straight line start -> best takeoff
    t0, _, _ = planner.plan(start, np.zeros((0, 2)), 6.0)
    mid = (start + t0) / 2
    t, c, _ = planner.plan(start, mid[None, :], 6.0)
    pts = [start, *([c] if c is not None else []), t]
    assert planner.clearance(pts, mid[None, :])[0] >= 1.5


def test_planner_is_fast():
    import time

    start = np.array([9.0, 4.0])
    d = np.array([[4.0, 1.0], [2.0, -1.0], [5.5, 3.0], [3.0, 2.5]])
    t0 = time.perf_counter()
    for _ in range(10):
        planner.plan(start, d, 6.0)
    assert (time.perf_counter() - t0) / 10 < 0.05
