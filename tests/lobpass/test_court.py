import numpy as np

from demos.lobpass import constants as C
from demos.lobpass import court


def test_ft_circle_is_a_semicircle_toward_half_court():
    (arc,) = court.line_polylines()["ft_circle"]
    centre = np.array([C.FT_LINE_X, 0.0])
    assert np.allclose(np.linalg.norm(arc - centre, axis=1), C.FT_CIRCLE_R)
    assert arc[:, 0].min() >= C.FT_LINE_X - 1e-9  # opens toward +x


def test_three_point_arc_meets_corner_lines():
    arc = court.line_polylines()["three"][1]
    assert np.allclose(np.linalg.norm(arc, axis=1), C.THREE_R)
    assert abs(abs(arc[0, 1]) - C.THREE_CORNER_Y) < 1e-9


def test_bounds_and_ring_distance():
    assert court.in_bounds(5, 0) and not court.in_bounds(13, 0) and not court.in_bounds(5, 8)
    assert abs(court.rim_ring_distance(C.RIM_RING_R, 0, C.RIM_TUBE_Z)) < 1e-12
    assert court.board_box_distance(C.BOARD_CENTER) == 0.0
