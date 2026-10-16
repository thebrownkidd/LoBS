"""SDF and MJCF fixtures."""

from __future__ import annotations

from pathlib import Path

SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <world name="w">
    <gravity>0 0 -9.81</gravity>
    <physics type="ode"><max_step_size>0.002</max_step_size></physics>
    <model name="ground">
      <static>true</static>
      <link name="link">
        <collision name="c">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <surface><friction><ode><mu>0.9</mu></ode></friction></surface>
        </collision>
      </link>
    </model>
    <model name="crate">
      <pose>1 0 0.5 0 0 0</pose>
      <link name="link">
        <inertial><mass>3</mass><inertia><ixx>0.08</ixx><iyy>0.08</iyy><izz>0.08</izz></inertia></inertial>
        <collision name="c">
          <geometry><box><size>0.4 0.4 0.4</size></box></geometry>
          <surface><friction><ode><mu>0.6</mu></ode></friction><bounce><restitution_coefficient>0.3</restitution_coefficient></bounce></surface>
        </collision>
        <visual name="v"><geometry><box><size>0.4 0.4 0.4</size></box></geometry></visual>
      </link>
    </model>
    <model name="pendulum">
      <pose>0 2 2 0 0 0</pose>
      <link name="arm">
        <pose>0 0 -0.5 0 0 0</pose>
        <inertial><mass>1</mass><inertia><ixx>0.09</ixx><iyy>0.09</iyy><izz>0.001</izz></inertia></inertial>
        <collision name="c"><geometry><cylinder><radius>0.03</radius><length>1.0</length></cylinder></geometry></collision>
      </link>
      <joint name="hinge" type="revolute">
        <parent>world</parent><child>arm</child><pose>0 0 0.5 0 0 0</pose>
        <axis><xyz>0 1 0</xyz><limit><lower>-0.8</lower><upper>0.8</upper><effort>20</effort></limit><dynamics><damping>0.02</damping></dynamics></axis>
      </joint>
    </model>
    <model name="shoulder_arm">
      <pose>2 2 2 0 0 0</pose>
      <link name="upper"><pose>0 0 -0.2 0 0 0</pose>
        <inertial><mass>0.8</mass><inertia><ixx>0.01</ixx><iyy>0.01</iyy><izz>0.001</izz></inertia></inertial>
        <collision name="c"><geometry><sphere><radius>0.05</radius></sphere></geometry></collision>
      </link>
      <joint name="ball" type="ball"><parent>world</parent><child>upper</child><pose>0 0 0.2 0 0 0</pose></joint>
    </model>
  </world>
</sdf>
"""

MJCF = """<mujoco model="demo">
  <compiler angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset><mesh name="tet" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/></asset>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" friction="0.8 0.005 0.0001"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" mass="2"/>
      <site name="top" pos="0 0 .1"/>
    </body>
    <body name="pend" pos="0 1 2">
      <joint name="hinge" type="hinge" axis="0 1 0" range="-30 60" damping=".01" armature=".002"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.5" size=".03" mass="1"/>
    </body>
    <body name="gimbal" pos="0 2 2">
      <joint name="gx" type="hinge" axis="1 0 0"/>
      <joint name="gy" type="hinge" axis="0 1 0"/>
      <geom type="sphere" size=".05" mass=".5"/>
    </body>
    <body name="shoulder" pos="1 1 2">
      <joint name="ball" type="ball" damping=".1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -.4" size=".03" mass=".8"/>
      <body name="forearm" pos="0 0 -.4">
        <joint name="elbow" type="hinge" axis="0 1 0" range="0 90"/>
        <geom type="capsule" fromto="0 0 0 0 0 -.3" size=".02" mass=".4"/>
      </body>
    </body>
    <body name="wedge" pos="-1 0 .5"><freejoint/><geom type="mesh" mesh="tet" mass="1"/></body>
  </worldbody>
  <actuator>
    <motor name="m_hinge" joint="hinge" ctrlrange="-5 5"/>
    <position name="p_elbow" joint="elbow" kp="20" kv="2"/>
  </actuator>
  <tendon><fixed name="t"><joint joint="gx" coef="1"/><joint joint="gy" coef="1"/></fixed></tendon>
</mujoco>
"""


def make_sdf(path: Path) -> Path:
    path.write_text(SDF, encoding="utf-8")
    return path


def make_mjcf(path: Path) -> Path:
    path.write_text(MJCF, encoding="utf-8")
    return path
