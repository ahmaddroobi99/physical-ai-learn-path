import numpy as np

from physical_ai_lab.camera import back_project, intrinsic_matrix, project_point
from physical_ai_lab.depth import depth_to_points, synthetic_depth
from physical_ai_lab.motion import estimate_shift
from physical_ai_lab.scene import make_scene


def test_projection_round_trip():
    intrinsics = intrinsic_matrix(640, 400, 520.0, 520.0)
    point = np.array([0.25, -0.10, 2.0], dtype=np.float64)
    u, v = project_point(point, intrinsics)
    recovered = back_project(u, v, float(point[2]), intrinsics)
    assert np.allclose(recovered, point)


def test_depth_cloud_keeps_positive_range():
    depth = synthetic_depth(160, 240)
    points = depth_to_points(depth, fx=220.0, fy=220.0, cx=120.0, cy=80.0)
    assert points.shape == (160 * 240, 3)
    assert np.all(points[:, 2] > 0)


def test_phase_correlation_recovers_pure_translation():
    frame0 = make_scene()
    frame1 = np.roll(frame0, shift=(3, 8), axis=(0, 1))
    dx, dy, response = estimate_shift(frame0, frame1)
    assert abs(dx - 8) < 1.0
    assert abs(dy - 3) < 1.0
    assert response > 0.2
