"""Estimate a 2D image translation with phase correlation."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_shift(frame0: np.ndarray, frame1: np.ndarray) -> tuple[float, float, float]:
    gray0 = np.float32(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY))
    gray1 = np.float32(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY))
    (dx, dy), response = cv2.phaseCorrelate(gray0, gray1)
    return float(dx), float(dy), float(response)
