"""The LoBS lob-pass demo: a passer learns to throw a lob to a running dunker through defenders."""

import os

# numpy (MKL) and torch each ship an OpenMP runtime; loading both in one process aborts on the first BLAS call.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
