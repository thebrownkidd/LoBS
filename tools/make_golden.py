"""Record golden histories from the current code: python tools/make_golden.py [name ...]"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.golden.scenarios import SCENARIOS, run  # noqa: E402

out = Path(__file__).resolve().parents[1] / "tests" / "golden"
for name in sys.argv[1:] or SCENARIOS:
    data = run(name)
    np.savez_compressed(out / f"{name}.npz", **data)
    print(f"{name}: {len(data)} arrays, {len(data['t'])} frames")
