"""Grayscale, Canny edges, and probabilistic Hough lines."""

from __future__ import annotations

import cv2
import numpy as np


def detect_edges_and_lines(
    image_bgr: np.ndarray,
    canny_low: int = 60,
    canny_high: int = 150,
    hough_threshold: int = 45,
    min_line_length: int = 25,
    max_line_gap: int = 8,
) -> dict[str, np.ndarray | int]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    line_view = image_bgr.copy()
    segments = _line_segments(lines)
    for x1, y1, x2, y2 in segments:
        cv2.line(line_view, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return {"gray": gray, "edges": edges, "line_view": line_view, "line_count": len(segments)}


def _line_segments(lines: np.ndarray | None) -> list[tuple[int, int, int, int]]:
    """OpenCV 4 returns (N, 1, 4). OpenCV 5 returns (N, 4)."""
    if lines is None:
        return []
    array = np.asarray(lines).reshape(-1, 4)
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in array]
