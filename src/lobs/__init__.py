"""The Laws of Bullshit: buildable physics worlds with anchors and measurements."""

__version__ = "0.1.0"

from . import backends, ops, props, rel, viz  # noqa: E402
from .check import check_world  # noqa: E402
from .compare import compare  # noqa: E402
from .measure.functions import register_function  # noqa: E402
from .ops import custom  # noqa: E402
from .schema import Box, Capsule, Cylinder, Path, Plane, Pose, Rigid, Sphere, TranslateOnly  # noqa: E402
from .sim import Simulation  # noqa: E402
from .trajectory import Trajectory  # noqa: E402
from .world import WORLD, Fixed, Hinge, Initial, Slide, World, load_world  # noqa: E402


def __getattr__(name):  # lazy: GymEnv needs the optional gymnasium dependency
    if name == "GymEnv":
        from .gym import GymEnv

        return GymEnv
    raise AttributeError(f"module 'lobs' has no attribute {name!r}")


def import_world(source, **kwargs):
    """Import a foreign world file (glTF, IFC, USD, URDF, SDF, MJCF, OBJ/STL) into a .lobs package.

    Returns ``(World, ImportReport)``. See ``lobs.importers.api.import_world``.
    """
    from .importers.api import import_world as _import_world

    return _import_world(source, **kwargs)


__all__ = [
    "WORLD", "Box", "Capsule", "Cylinder", "Fixed", "Hinge", "Initial", "Path", "Plane", "Pose",
    "Rigid", "Simulation", "Slide", "Sphere", "Trajectory", "TranslateOnly", "World", "__version__",
    "GymEnv", "backends", "check_world", "compare", "custom", "import_world", "load_world", "ops", "props", "register_function", "rel", "viz",
]  # fmt: skip
