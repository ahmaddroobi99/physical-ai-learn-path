"""Pinhole camera projection and its inverse."""

from __future__ import annotations

import numpy as np


def intrinsic_matrix(width: int, height: int, fx: float, fy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_point(point_camera: np.ndarray, intrinsics: np.ndarray) -> tuple[float, float]:
    x_m, y_m, z_m = (float(value) for value in point_camera)
    if z_m == 0.0:
        raise ValueError("depth Z must be non-zero")
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    u = fx * x_m / z_m + cx
    v = fy * y_m / z_m + cy
    return u, v


def back_project(u: float, v: float, depth_m: float, intrinsics: np.ndarray) -> np.ndarray:
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x_m = (u - cx) * depth_m / fx
    y_m = (v - cy) * depth_m / fy
    return np.array([x_m, y_m, depth_m], dtype=np.float64)
