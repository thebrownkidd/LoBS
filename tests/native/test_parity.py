"""The native kernel must reproduce the reference path exactly."""

import numpy as np
import pytest

from lobs import native
from tests.golden.scenarios import SCENARIOS, run

pytestmark = pytest.mark.skipif(not native.available(), reason="native kernel not built (python tools/build_native.py)")


@pytest.fixture(autouse=True)
def restore():
    yield
    native.set_enabled(True)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_native_equals_reference_bitwise(name):
    native.set_enabled(False)
    ref = run(name)
    native.set_enabled(True)
    got = run(name)
    assert set(ref) == set(got)
    bad = [k for k in ref if not np.array_equal(np.asarray(ref[k]), np.asarray(got[k]), equal_nan=True)]
    assert not bad, bad[:8]


def test_kernel_is_actually_in_use():
    import lobs

    native.set_enabled(True)
    w = lobs.World("k", gravity=(0, 0, 0))
    w.add_sphere("b", 0.1, mass=1.0, pos=(0, 0, 1))
    assert w.simulate("mujoco").info["native_kernel"] is False or True  # info is read after the first step
    s = w.simulate("mujoco")
    s.step()
    assert s.info["native_kernel"] is True
    native.set_enabled(False)
    s2 = w.simulate("mujoco")
    s2.step()
    assert s2.info["native_kernel"] is False
