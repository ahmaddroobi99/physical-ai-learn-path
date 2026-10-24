"""Load the lab settings shared by every lesson."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


def project_root() -> Path:
    """Return the repository root, whether the package is imported from src or an editable install."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs" / "lab.json").is_file():
            return parent
    return here.parents[2]


@dataclass(frozen=True)
class LabConfig:
    seed: int = 7
    width: int = 640
    height: int = 400
    fx: float = 520.0
    fy: float = 520.0
    canny_low: int = 60
    canny_high: int = 150
    hough_threshold: int = 45
    hough_min_line_length: int = 25
    hough_max_line_gap: int = 8
    orb_features: int = 1000
    pipeline_orb_features: int = 500
    match_ratio: float = 0.75
    detection_threshold: float = 0.5
    quant_scale: float = 32.0
    benchmark_frames: int = 20
    scene_shift_x: int = 8
    scene_shift_y: int = 3


def load_config(path: Path | None = None) -> LabConfig:
    config_path = path or (project_root() / "configs" / "lab.json")
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    known = {item.name for item in fields(LabConfig)}
    return LabConfig(**{key: raw[key] for key in known if key in raw})
