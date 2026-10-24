"""Array helpers used before any vision code."""

from __future__ import annotations

import numpy as np

SENSOR_READINGS = np.array([1.2, 1.4, 1.1, 1.8, 2.0], dtype=np.float32)
DISTANCES_M = np.array([0.7, 1.4, 2.3, 4.0], dtype=np.float32)


def normalize(values: np.ndarray) -> np.ndarray:
    """Scale values into [0, 1]. A constant input becomes zeros."""
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    lo = float(array.min())
    hi = float(array.max())
    if hi == lo:
        return np.zeros_like(array)
    return (array - lo) / (hi - lo)


def meters_to_centimeters(distances_m: np.ndarray) -> np.ndarray:
    return np.asarray(distances_m, dtype=np.float32) * np.float32(100.0)


def meters_per_second_to_kmh(speed_mps: np.ndarray) -> np.ndarray:
    return np.asarray(speed_mps, dtype=np.float32) * np.float32(3.6)
