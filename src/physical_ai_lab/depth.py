"""Back-project a depth map into a camera-frame point cloud."""

from __future__ import annotations

import numpy as np


def synthetic_depth(height: int = 160, width: int = 240) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    return (2.0 + 0.003 * xx + 0.002 * yy).astype(np.float64)


def depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[0:height, 0:width]
    z_m = np.asarray(depth, dtype=np.float64)
    x_m = (xx - cx) * z_m / fx
    y_m = (yy - cy) * z_m / fy
    points = np.stack([x_m, y_m, z_m], axis=-1).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    return points[valid]
