"""Synthetic scene so the lessons do not depend on a downloaded dataset."""

from __future__ import annotations

import cv2
import numpy as np


def make_scene(
    width: int = 640,
    height: int = 400,
    shift_x: int = 0,
    shift_y: int = 0,
) -> np.ndarray:
    """Build a BGR image with a grid and high-contrast shapes that features can track."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (24, 28, 34)
    for x in range(20, width, 40):
        cv2.line(image, (x + shift_x, 0), (x + shift_x, height - 1), (55, 60, 67), 1)
    for y in range(20, height, 40):
        cv2.line(image, (0, y + shift_y), (width - 1, y + shift_y), (55, 60, 67), 1)
    for i in range(12):
        x = 50 + (i * 47) % (width - 100)
        y = 45 + (i * 31) % (height - 90)
        x = int((x + shift_x) % width)
        y = int((y + shift_y) % height)
        cv2.rectangle(
            image,
            (x, y),
            (min(x + 25, width - 1), min(y + 18, height - 1)),
            (70 + (i * 13) % 180, 120 + (i * 7) % 120, 220),
            -1,
        )
        cv2.circle(
            image,
            (min(x + 34, width - 1), min(y + 27, height - 1)),
            5,
            (240, 240, 240),
            -1,
        )
    cv2.putText(
        image,
        "PHYSICAL AI LAB",
        (22, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (230, 230, 230),
        2,
    )
    return image


def crop_and_resize(image: np.ndarray) -> dict[str, np.ndarray]:
    """Show that a crop and a resize are views of the same array data with different shapes."""
    crop = image[40:240, 60:360]
    small = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    return {"crop": crop, "half": small}
