import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from lobs import inertia as I
from lobs import math3d as m3
from lobs.schema import Box, Capsule, Cylinder, Sphere

vec = st.tuples(*[st.floats(-5, 5)] * 3)
nonzero = vec.filter(lambda v: np.linalg.norm(v) > 0.1)


@given(nonzero, st.floats(-6, 6))
def test_axis_angle_roundtrip(axis, ang):
    q = m3.quat_from_axis_angle(axis, ang)
    r = m3.quat_to_mat(q)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.allclose(m3.quat_to_mat(m3.mat_to_quat(r)), r, atol=1e-9)
    assert np.allclose(m3.quat_exp(m3.quat_log(q)), q, atol=1e-9) or np.allclose(
        m3.quat_exp(m3.quat_log(q)), -q, atol=1e-9
    )


@given(nonzero, st.floats(-3, 3), nonzero, st.floats(-3, 3), vec)
def test_mul_matches_matrix(a1, t1, a2, t2, v):
    q1, q2 = m3.quat_from_axis_angle(a1, t1), m3.quat_from_axis_angle(a2, t2)
    lhs = m3.quat_rotate(m3.quat_mul(q1, q2), v)
    rhs = m3.quat_rotate(q1, m3.quat_rotate(q2, v))
    assert np.allclose(lhs, rhs, atol=1e-9)


def test_double_cover():
    q = m3.quat_from_axis_angle([0, 0, 1], 0.7)
    assert m3.quat_same_rotation(q, -q)
    assert abs(m3.quat_angle(-q) - 0.7) < 1e-12


@given(st.floats(-50, 50))
def test_wrap_angle_range(a):
    w = float(m3.wrap_angle(a))
    assert -np.pi < w <= np.pi + 1e-12
    assert abs(np.sin(w) - np.sin(a)) < 1e-9 and abs(np.cos(w) - np.cos(a)) < 1e-9


def test_wrap_edges():
    assert np.isclose(m3.wrap_angle(np.pi), np.pi)
    assert np.isclose(m3.wrap_angle(-np.pi), np.pi)


def test_angle_between_degenerate_and_small():
    assert np.isnan(m3.angle_between([0, 0, 0], [1, 0, 0]))
    assert np.isclose(m3.angle_between([1, 0, 0], [1, 1e-9, 0]), 1e-9, rtol=1e-6)
    assert np.isclose(m3.angle_between([1, 0, 0], [-1, 0, 0]), np.pi)


def test_quat_exp_small():
    q = m3.quat_exp(np.array([1e-15, 0, 0]))
    assert np.allclose(q, [1, 5e-16, 0, 0])


def test_box_inertia():
    i = I.shape_inertia(Box(size=(1, 2, 3)), 12.0)
    assert np.allclose(np.diag(i), [13, 10, 5])


def test_capsule_inertia_matches_sphere_when_no_cylinder():
    r = 0.3
    i = I.shape_inertia(Capsule(radius=r, length=0.0), 2.0)
    assert np.allclose(np.diag(i), 0.4 * 2.0 * r * r)


def test_capsule_vs_numeric_integration():
    r, L, m = 0.2, 0.7, 1.5
    rng = np.random.default_rng(0)
    n = 400000
    pts = rng.uniform([-r, -r, -(L / 2 + r)], [r, r, L / 2 + r], size=(n, 3))
    z = np.clip(pts[:, 2], -L / 2, L / 2)
    inside = (pts[:, 0] ** 2 + pts[:, 1] ** 2 + (pts[:, 2] - z) ** 2) <= r * r
    p = pts[inside]
    dm = m / len(p)
    ixx = dm * np.sum(p[:, 1] ** 2 + p[:, 2] ** 2)
    izz = dm * np.sum(p[:, 0] ** 2 + p[:, 1] ** 2)
    i = I.shape_inertia(Capsule(radius=r, length=L), m)
    assert abs(i[0, 0] - ixx) / ixx < 0.02
    assert abs(i[2, 2] - izz) / izz < 0.02


def test_combine_parallel_axis():
    a = (1.0, [1, 0, 0], np.zeros((3, 3)))
    b = (1.0, [-1, 0, 0], np.zeros((3, 3)))
    mass, com, inertia = I.combine([a, b])
    assert mass == 2 and np.allclose(com, 0)
    assert np.allclose(np.diag(inertia), [0, 2, 2])


@settings(max_examples=30)
@given(st.tuples(*[st.floats(0.1, 3)] * 3), nonzero, st.floats(-3, 3))
def test_principal_axes_reconstruct(size, axis, ang):
    r = m3.quat_to_mat(m3.quat_from_axis_angle(axis, ang))
    inertia = r @ I.shape_inertia(Box(size=size), 2.0) @ r.T
    d, q = I.principal_axes(inertia)
    rp = m3.quat_to_mat(q)
    assert np.linalg.det(rp) > 0
    assert np.allclose(rp @ np.diag(d) @ rp.T, inertia, atol=1e-9)


def test_volumes():
    assert np.isclose(I.shape_volume(Sphere(radius=1)), 4 / 3 * np.pi)
    assert np.isclose(I.shape_volume(Cylinder(radius=1, length=2)), 2 * np.pi)
