import numpy as np
import pytest

import lobs
from lobs.gui.core.gust import gust_plan, inject_gust


def test_plan_respects_peak_and_delivers_exact_impulse():
    ticks, mag = gust_plan(peak=10.0, impulse=1.0, tick_dt=0.01)
    assert mag <= 10.0 and mag * ticks * 0.01 == pytest.approx(1.0)
    ticks, mag = gust_plan(peak=100.0, impulse=0.05, tick_dt=0.01)
    assert ticks == 1 and mag == pytest.approx(5.0)


@pytest.mark.parametrize("be", lobs.backends.available())
def test_gust_changes_momentum_by_the_budget(be):
    w = lobs.World("g", dt=0.002, substeps=5, gravity=(0, 0, 0))
    w.add_box("b", (0.2, 0.2, 0.2), mass=2.0, pos=(0, 0, 1))
    sim = w.simulate(be)
    info = inject_gust(sim, "b", (1.0, 0.0, 0.0), peak=8.0, impulse=1.0)
    sim.run(seconds=1.0)
    assert info["impulse"] == pytest.approx(1.0)
    v = sim.state().body_linvel[0]
    assert v[0] == pytest.approx(0.5, rel=2e-3) and np.allclose(v[1:], 0, atol=1e-6)  # J / m
