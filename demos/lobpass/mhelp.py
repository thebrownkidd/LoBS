"""Small measure-building helpers shared by world, machines and rules. Everything here stays inside the native
measure interpreter's op set, so worlds built from it keep running on the native tick core."""

from __future__ import annotations

from lobs import ops, props

FLAT = ops.vec(1, 1, 0)


def dimless(m):
    return ops.as_unit(m, "dimensionless")


def newton(m):
    return ops.as_unit(m, "newton")


def body_path(player: str) -> str:
    return f"{player}/body"


def pos_of(player: str):
    """Player position as a unit-free 3-vector: (x, y, jz), the body origin sits at floor level."""
    return dimless(props.position(f"{body_path(player)}:origin"))


def vel_of(player: str):
    return dimless(props.velocity(f"{body_path(player)}:origin"))


def ball_pos():
    return dimless(props.position("ball"))


def ball_vel():
    return dimless(props.velocity("ball"))


def var(beh: str, name: str):
    """A behavior variable as a unit-free measure (scalar or 3-vector)."""
    return dimless(props.behavior_var(beh, name))


def always():
    return dimless(ops.time()) >= 0.0
