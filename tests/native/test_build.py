import numpy as np
import pytest

import lobs
from lobs import native

pytestmark = pytest.mark.skipif(not native.available(), reason="native kernel not built (python tools/build_native.py)")


def test_native_module_loads_and_matches_mujoco_version():
    import mujoco

    n = native.module()
    assert n.init_mujoco().startswith(mujoco.mj_versionString())


def test_native_mj_step_is_bitwise_identical_to_python():
    import mujoco

    w = lobs.World("n", dt=0.002, substeps=5)
    w.add_box("floor", (10, 10, 0.2), fixed=True, pos=(0, 0, -0.1))
    w.add_sphere("ball", 0.1, mass=0.5, pos=(0, 0, 1.0), linvel=(1.0, 0.3, 2.0), restitution=0.7)
    a, b = w.simulate("mujoco", record=False), w.simulate("mujoco", record=False)
    (ma, da), (mb, db) = a.native, b.native
    for _ in range(300):
        mujoco.mj_step(ma, da, 5)
    native.module().step_raw(mb._address, db._address, 1500)
    assert np.array_equal(da.qpos, db.qpos) and np.array_equal(da.qvel, db.qvel)
