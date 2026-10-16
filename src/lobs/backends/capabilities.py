from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Capability(StrEnum):
    RIGID_BODIES = "rigid_bodies"
    SHAPE_BOX = "shape_box"
    SHAPE_SPHERE = "shape_sphere"
    SHAPE_CYLINDER = "shape_cylinder"
    SHAPE_CAPSULE = "shape_capsule"
    SHAPE_PLANE = "shape_plane"
    SHAPE_CONVEX_MESH = "shape_convex_mesh"
    SHAPE_CONVEX_SET = "shape_convex_set"
    SHAPE_TRIANGLE_MESH_STATIC = "shape_triangle_mesh_static"
    SHAPE_HEIGHTFIELD = "shape_heightfield"
    JOINT_HINGE = "joint_hinge"
    JOINT_SLIDE = "joint_slide"
    JOINT_FIXED = "joint_fixed"
    JOINT_BALL = "joint_ball"
    JOINT_LIMITS = "joint_limits"
    JOINT_DAMPING = "joint_damping"
    JOINT_ARMATURE = "joint_armature"
    ACTUATOR_EFFORT = "actuator_effort"
    ACTUATOR_POSITION = "actuator_position"
    ACTUATOR_VELOCITY = "actuator_velocity"
    FRICTION = "friction"
    RESTITUTION = "restitution"
    COLLISION_GROUPS = "collision_groups"
    BODY_ACCELERATION = "body_acceleration"
    JOINT_ACCELERATION = "joint_acceleration"
    CONTACTS = "contacts"
    CONTACT_FORCES = "contact_forces"
    EXTERNAL_WRENCH = "external_wrench"
    NATIVE_MJCF = "native_mjcf"
    DETERMINISTIC_RESET = "deterministic_reset"


@dataclass(frozen=True)
class Support:
    level: str  # 'native' | 'approximate' | 'unsupported'
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.level != "unsupported"


NATIVE = Support("native")
UNSUPPORTED = Support("unsupported")


def APPROXIMATE(note: str) -> Support:  # noqa: N802
    return Support("approximate", note)


SHAPE_CAP = {
    "box": Capability.SHAPE_BOX,
    "sphere": Capability.SHAPE_SPHERE,
    "cylinder": Capability.SHAPE_CYLINDER,
    "capsule": Capability.SHAPE_CAPSULE,
    "plane": Capability.SHAPE_PLANE,
    "convex_mesh": Capability.SHAPE_CONVEX_MESH,
    "convex_set": Capability.SHAPE_CONVEX_SET,
    "triangle_mesh": Capability.SHAPE_TRIANGLE_MESH_STATIC,
    "heightfield": Capability.SHAPE_HEIGHTFIELD,
}
JOINT_CAP = {
    "hinge": Capability.JOINT_HINGE,
    "slide": Capability.JOINT_SLIDE,
    "fixed": Capability.JOINT_FIXED,
    "ball": Capability.JOINT_BALL,
}
ACTUATOR_CAP = {
    "effort": Capability.ACTUATOR_EFFORT,
    "position": Capability.ACTUATOR_POSITION,
    "velocity": Capability.ACTUATOR_VELOCITY,
}
