import os

os.environ.setdefault(
    "KMP_DUPLICATE_LIB_OK", "TRUE"
)  # numpy and torch OpenMP runtimes in one process
