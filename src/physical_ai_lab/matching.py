"""ORB keypoints and Lowe's ratio test between two frames."""

from __future__ import annotations

import cv2
import numpy as np


def match_orb(
    frame0: np.ndarray,
    frame1: np.ndarray,
    nfeatures: int = 1000,
    ratio: float = 0.75,
    draw_limit: int = 60,
) -> dict[str, object]:
    orb = cv2.ORB_create(nfeatures=nfeatures)
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    keypoints0, descriptors0 = orb.detectAndCompute(gray0, None)
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)

    matches: list[cv2.DMatch] = []
    if descriptors0 is not None and descriptors1 is not None and len(keypoints0) > 1 and len(keypoints1) > 1:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        pairs = matcher.knnMatch(descriptors0, descriptors1, k=2)
        matches = [first for first, second in pairs if first.distance < ratio * second.distance]

    matched_view = cv2.drawMatches(
        frame0,
        keypoints0,
        frame1,
        keypoints1,
        matches[:draw_limit],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return {
        "keypoints0": keypoints0,
        "keypoints1": keypoints1,
        "matches": matches,
        "matched_view": matched_view,
    }
