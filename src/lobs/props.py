"""World-relative properties of an actor, body, anchor or joint.

Each function returns a ``Measure``; add it to a world with ``world.measure(name, m)``.
Targets may be handles or path strings (``"box"``, ``"arm/hand"``, ``"box/top"``, ``"box:com"``).
"""

from __future__ import annotations

from .measure.nodes import Measure
from .schema.measure import PropNode


def _path(x) -> str:
    return str(getattr(x, "path", x))


def _p(prop: str, target, **args) -> Measure:
    return Measure(PropNode(prop=prop, target=_path(target), args=args))


def position(target):
    return _p("position", target)  # noqa: E704


def height(target):
    return _p("height", target)  # noqa: E704


def orientation(target):
    return _p("orientation", target)  # noqa: E704


def tilt(target):
    return _p("tilt", target)  # noqa: E704


def heading(target):
    return _p("heading", target)  # noqa: E704


def velocity(target):
    return _p("velocity", target)  # noqa: E704


def speed(target):
    return _p("speed", target)  # noqa: E704


def angular_velocity(target):
    return _p("angular_velocity", target)  # noqa: E704


def spin(target):
    return _p("spin", target)  # noqa: E704


def acceleration(target):
    return _p("acceleration", target)  # noqa: E704


def displacement(target):
    return _p("displacement", target)  # noqa: E704


def path_length(target):
    return _p("path_length", target)  # noqa: E704


def kinetic_energy(target):
    return _p("kinetic_energy", target)  # noqa: E704


def potential_energy(target):
    return _p("potential_energy", target)  # noqa: E704


def mechanical_energy(target):
    return _p("mechanical_energy", target)  # noqa: E704


def momentum(target):
    return _p("momentum", target)  # noqa: E704


def angular_momentum(target):
    return _p("angular_momentum", target)  # noqa: E704


def mass(target):
    return _p("mass", target)  # noqa: E704


def in_contact(target, with_=None):
    return _p("in_contact", target, **({"with": _path(with_)} if with_ is not None else {}))


def contact_count(target, with_=None):
    return _p("contact_count", target, **({"with": _path(with_)} if with_ is not None else {}))


def contact_force(target, with_=None):
    return _p("contact_force", target, **({"with": _path(with_)} if with_ is not None else {}))


def joint_position(joint):
    return _p("joint_position", joint)  # noqa: E704


def joint_velocity(joint):
    return _p("joint_velocity", joint)  # noqa: E704


def joint_effort(joint):
    return _p("joint_effort", joint)  # noqa: E704


def joint_limit_margin(joint):
    return _p("joint_limit_margin", joint)  # noqa: E704
