"""Compare a Python loop with a vectorized NumPy reduction."""

from __future__ import annotations

import time

import numpy as np


def python_square_sum(values: np.ndarray) -> float:
    total = 0.0
    for value in values:
        number = float(value)
        total += number * number
    return total


def time_square_sums(values: np.ndarray, loop_count: int = 100_000) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float32)
    started = time.perf_counter()
    numpy_result = float(np.sum(array * array))
    numpy_seconds = time.perf_counter() - started

    subset = array[:loop_count]
    started = time.perf_counter()
    loop_result = python_square_sum(subset)
    loop_seconds = time.perf_counter() - started

    return {
        "numpy_result": numpy_result,
        "numpy_ms": numpy_seconds * 1000.0,
        "numpy_count": float(array.size),
        "loop_result": loop_result,
        "loop_ms": loop_seconds * 1000.0,
        "loop_count": float(subset.size),
    }
