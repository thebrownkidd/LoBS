"""Optimised code must reproduce the recorded histories (tools/make_golden.py records them)."""

from pathlib import Path

import numpy as np
import pytest

from tests.golden.scenarios import SCENARIOS, run

GOLDEN = Path(__file__).parent / "golden"
RTOL, ATOL = 1e-9, 1e-11


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_matches_golden(name):
    ref = np.load(GOLDEN / f"{name}.npz")
    got = run(name)
    assert set(got) == set(ref.files)
    worst = []
    for k in ref.files:
        a, b = np.asarray(got[k], dtype=float), np.asarray(ref[k], dtype=float)
        assert a.shape == b.shape, k
        ok = np.allclose(a, b, rtol=RTOL, atol=ATOL, equal_nan=True)
        if not ok:
            worst.append((k, float(np.nanmax(np.abs(a - b)))))
    assert not worst, f"differs from golden: {sorted(worst, key=lambda x: -x[1])[:6]}"


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_runs_are_bitwise_deterministic(name):
    a, b = run(name), run(name)
    for k in a:
        assert np.array_equal(np.asarray(a[k]), np.asarray(b[k]), equal_nan=True), k
