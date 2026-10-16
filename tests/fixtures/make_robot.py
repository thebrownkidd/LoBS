"""Robot description fixtures (URDF / SDF / MJCF), generated as text at test time."""

from __future__ import annotations

from pathlib import Path

URDF = """<?xml version="1.0"?>
<robot name="arm">
  <material name="blue"><color rgba="0.1 0.2 0.9 1"/></material>
  <link name="base">
    <inertial><origin xyz="0 0 0.05"/><mass value="5"/><inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.05"/></inertial>
    <visual><geometry><box size="0.2 0.2 0.1"/></geometry><material name="blue"/></visual>
    <collision><origin xyz="0 0 0.05"/><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="upper">
    <inertial>
      <origin xyz="0 0 0.25" rpy="0 0 0.5"/><mass value="2"/>
      <inertia ixx="0.1" ixy="0.01" ixz="0" iyy="0.2" iyz="0" izz="0.05"/>
    </inertial>
    <visual><origin xyz="0 0 0.25"/><geometry><cylinder radius="0.04" length="0.5"/></geometry></visual>
    <collision><origin xyz="0 0 0.25"/><geometry><capsule radius="0.04" length="0.42"/></geometry></collision>
  </link>
  <link name="slider">
    <inertial><origin xyz="0 0 0"/><mass value="1"/><inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <visual><geometry><mesh filename="package://robot_pkg/meshes/block.stl" scale="1 1 1"/></geometry></visual>
    <collision><geometry><mesh filename="package://robot_pkg/meshes/block.stl" scale="1 1 1"/></geometry></collision>
  </link>
  <link name="wheel">
    <inertial><origin xyz="0 0.01 0" rpy="0 0 0.5"/><mass value="0.5"/><inertia ixx="0.005" ixy="0.001" ixz="0" iyy="0.004" iyz="0" izz="0.003"/></inertial>
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <link name="ghost"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="upper"/><origin xyz="0 0 0.1" rpy="0 0 1.5707963"/><axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.2" effort="30" velocity="2"/><dynamics damping="0.01"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="upper"/><child link="slider"/><origin xyz="0 0 0.5"/><axis xyz="0 0 1"/>
    <limit lower="-0.1" upper="0.2" effort="50" velocity="1"/>
  </joint>
  <joint name="spin" type="continuous">
    <parent link="slider"/><child link="wheel"/><origin xyz="0.1 0 0"/><axis xyz="0 1 0"/>
  </joint>
  <joint name="mount" type="fixed"><parent link="base"/><child link="ghost"/><origin xyz="0.3 0 0"/></joint>
</robot>
"""


def make_urdf(root: Path, *, with_mesh=True) -> tuple[Path, dict]:
    """Writes robot_pkg/{urdf/arm.urdf, meshes/block.stl}; returns (urdf path, import kwargs)."""
    import trimesh

    pkg = root / "robot_pkg"
    (pkg / "urdf").mkdir(parents=True, exist_ok=True)
    (pkg / "meshes").mkdir(exist_ok=True)
    if with_mesh:
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(pkg / "meshes" / "block.stl")
    path = pkg / "urdf" / "arm.urdf"
    path.write_text(URDF, encoding="utf-8")
    return path, {"package_dirs": {"robot_pkg": str(pkg)}}
