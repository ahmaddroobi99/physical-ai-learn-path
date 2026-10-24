"""One frame through grayscale, edges, and ORB, with per-stage timings."""

from __future__ import annotations

import time

import cv2
import numpy as np


def process_frame(
    image: np.ndarray,
    nfeatures: int = 500,
    canny_low: int = 60,
    canny_high: int = 150,
) -> dict[str, object]:
    timings: dict[str, float] = {}

    started = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    timings["grayscale_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), canny_low, canny_high)
    timings["edges_ms"] = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    keypoints, descriptors = cv2.ORB_create(nfeatures=nfeatures).detectAndCompute(gray, None)
    timings["orb_ms"] = (time.perf_counter() - started) * 1000.0

    return {
        "gray": gray,
        "edges": edges,
        "keypoints": keypoints,
        "descriptors": descriptors,
        "timings_ms": timings,
    }


def benchmark_frames(frames: list[np.ndarray], nfeatures: int = 500) -> dict[str, object]:
    records = [process_frame(frame, nfeatures=nfeatures)["timings_ms"] for frame in frames]
    if not records:
        raise ValueError("benchmark_frames requires at least one frame")
    stages: dict[str, dict[str, float]] = {}
    for stage in records[0]:
        values = np.array([record[stage] for record in records], dtype=np.float64)
        stages[stage] = {
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
        }
    totals = np.array([sum(record.values()) for record in records], dtype=np.float64)
    median_total_ms = float(np.median(totals))
    frames_per_second = 1000.0 / median_total_ms if median_total_ms > 0.0 else float("inf")
    return {
        "stages": stages,
        "median_total_ms": median_total_ms,
        "frames_per_second": frames_per_second,
    }
