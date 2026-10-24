"""Lesson catalog for the interactive lab. Each lesson calls one real function."""

from __future__ import annotations

import base64
import io
import platform
import sys
import time
from typing import Callable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physical_ai_lab import __version__
from physical_ai_lab.camera import back_project, intrinsic_matrix, project_point
from physical_ai_lab.depth import depth_to_points, synthetic_depth
from physical_ai_lab.edges import detect_edges_and_lines
from physical_ai_lab.foundations import meters_per_second_to_kmh, meters_to_centimeters, normalize
from physical_ai_lab.matching import match_orb
from physical_ai_lab.motion import estimate_shift
from physical_ai_lab.pipeline import benchmark_frames, process_frame
from physical_ai_lab.profiling import time_square_sums
from physical_ai_lab.quantize import quantize_int8
from physical_ai_lab.scene import crop_and_resize, make_scene

Runner = Callable[[dict], dict]


def _png_bgr(image: np.ndarray) -> str:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode the image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _png_figure(figure: plt.Figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _image(title: str, png: str) -> dict[str, str]:
    return {"title": title, "png": png}


def _numbers(text: str, limit: int) -> np.ndarray:
    pieces = [piece.strip() for piece in str(text).replace(";", ",").split(",") if piece.strip()]
    if not pieces:
        raise ValueError("Enter at least one number, separated by commas.")
    if len(pieces) > limit:
        raise ValueError(f"Use at most {limit} numbers.")
    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("Every entry must be a number, for example 1.2, 1.4, 1.8.") from exc
    return np.asarray(values, dtype=np.float32)


def _clamp_number(value: object, lower: float, upper: float, as_int: bool) -> float | int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("A slider value must be a number.") from exc
    number = min(upper, max(lower, number))
    if as_int:
        return int(round(number))
    return number


def run_environment(_: dict) -> dict:
    import torch

    lines = [
        f"physical-ai-lab {__version__}",
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"NumPy: {np.__version__}",
        f"OpenCV: {cv2.__version__}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        lines.append("PyTorch will use the CPU for the matrix lesson and the detector.")
    return {
        "call": "environment_check()",
        "lines": lines,
        "images": [],
        "notice": "These are the tools the later lessons call. A missing GPU does not block the path.",
    }


def run_foundations(params: dict) -> dict:
    readings = _numbers(params["readings"], limit=12)
    distances = _numbers(params["distances_m"], limit=12)
    speed = np.array([float(params["speed_mps"])], dtype=np.float32)
    scaled = normalize(readings)
    centimeters = meters_to_centimeters(distances)
    kmh = meters_per_second_to_kmh(speed)
    same_reading = float(readings.min()) == float(readings.max())
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.1))
    axes[0].bar(range(len(readings)), readings, color="#c4b29a")
    axes[0].set_title("Raw readings")
    axes[0].set_xlabel("index")
    axes[1].bar(range(len(scaled)), scaled, color="#b4532a")
    axes[1].set_title("Normalized, 0 to 1")
    axes[1].set_xlabel("index")
    axes[1].set_ylim(0, 1.05)
    figure.tight_layout()
    reading_text = ", ".join(f"{value:.4g}" for value in readings)
    return {
        "call": (
            f"normalize([{reading_text}])\n"
            f"meters_to_centimeters([{', '.join(f'{value:.4g}' for value in distances)}])\n"
            f"meters_per_second_to_kmh([{float(speed[0]):.4g}])"
        ),
        "lines": [
            f"Raw readings: {np.array2string(readings, precision=4)}",
            f"Normalized: {np.array2string(scaled, precision=4)}",
            f"Centimeters: {np.array2string(centimeters, precision=4)}",
            f"{float(speed[0]):.4g} m/s = {float(kmh[0]):.4g} km/h",
        ],
        "images": [_image("Raw readings and the normalized copy", _png_figure(figure))],
        "notice": (
            "Every reading is the same, so normalize returns zeros. "
            if same_reading
            else "The smallest reading sits at 0 and the largest sits at 1. "
        )
        + "Centimeters are meters times 100. Kilometers per hour are meters per second times 3.6.",
    }


def run_scene(params: dict) -> dict:
    frame0 = make_scene(params["width"], params["height"])
    frame1 = make_scene(params["width"], params["height"], params["shift_x"], params["shift_y"])
    views = crop_and_resize(frame0)
    return {
        "call": (
            f"make_scene(width={params['width']}, height={params['height']})\n"
            f"make_scene(width={params['width']}, height={params['height']}, "
            f"shift_x={params['shift_x']}, shift_y={params['shift_y']})"
        ),
        "lines": [
            f"Image shape: {frame0.shape}  (height, width, colors)",
            f"Bytes: {frame0.nbytes:,}",
            f"Crop shape: {views['crop'].shape}",
            f"Half-size shape: {views['half'].shape}",
        ],
        "images": [
            _image("Frame 0", _png_bgr(frame0)),
            _image("Shifted drawing", _png_bgr(frame1)),
            _image("Crop of frame 0", _png_bgr(views["crop"])),
        ],
        "notice": (
            "Three numbers per pixel are blue, green, and red. "
            "The crop is a smaller window of the same picture. The half-size copy has a new shape."
        ),
    }


def run_profiling(params: dict) -> dict:
    values = np.random.default_rng(7).random(params["count"], dtype=np.float32)
    timed = time_square_sums(values, loop_count=params["loop_count"])
    figure, axis = plt.subplots(figsize=(6.4, 3.2))
    labels = [
        f"NumPy\n{int(timed['numpy_count']):,} values",
        f"Python loop\n{int(timed['loop_count']):,} values",
    ]
    axis.bar(labels, [timed["numpy_ms"], timed["loop_ms"]], color=["#1c1915", "#b4532a"])
    axis.set_ylabel("milliseconds")
    axis.set_title("Time to add the squares")
    figure.tight_layout()
    return {
        "call": f"time_square_sums(values, loop_count={params['loop_count']})  # {params['count']:,} values",
        "lines": [
            f"NumPy: {timed['numpy_ms']:.2f} ms for {int(timed['numpy_count']):,} values",
            f"Python loop: {timed['loop_ms']:.2f} ms for {int(timed['loop_count']):,} values",
            f"NumPy square sum: {timed['numpy_result']:.4f}",
            f"Loop square sum: {timed['loop_result']:.4f}",
        ],
        "images": [_image("NumPy time beside the Python loop", _png_figure(figure))],
        "notice": (
            "The loop visits one number at a time. NumPy works on the whole array. "
            "The two sums match when they see the same values."
        ),
    }


def run_edges(params: dict) -> dict:
    low = min(params["canny_low"], params["canny_high"])
    high = max(params["canny_low"], params["canny_high"])
    frame = make_scene(params["width"], params["height"])
    found = detect_edges_and_lines(
        frame,
        canny_low=low,
        canny_high=high,
        hough_threshold=params["hough_threshold"],
        min_line_length=params["min_line_length"],
        max_line_gap=params["max_line_gap"],
    )
    return {
        "call": (
            "detect_edges_and_lines(frame, "
            f"canny_low={low}, canny_high={high}, "
            f"hough_threshold={params['hough_threshold']}, "
            f"min_line_length={params['min_line_length']}, "
            f"max_line_gap={params['max_line_gap']})"
        ),
        "lines": (
            ["Canny low was above Canny high, so the two thresholds were swapped."]
            if low != params["canny_low"]
            else []
        )
        + [f"Detected lines: {found['line_count']}"],
        "images": [
            _image("Grayscale", _png_bgr(found["gray"])),
            _image("Canny edges", _png_bgr(found["edges"])),
            _image("Lines drawn back on the picture", _png_bgr(found["line_view"])),
        ],
        "notice": (
            "Lower Canny thresholds keep fainter brightness changes. "
            "A higher Hough threshold keeps only the stronger straight lines."
        ),
    }


def run_matching(params: dict) -> dict:
    frame0 = make_scene(params["width"], params["height"])
    frame1 = make_scene(params["width"], params["height"], params["shift_x"], params["shift_y"])
    matched = match_orb(frame0, frame1, nfeatures=params["nfeatures"], ratio=params["ratio"])
    return {
        "call": (
            "match_orb(frame0, frame1, "
            f"nfeatures={params['nfeatures']}, ratio={params['ratio']:.2f})"
        ),
        "lines": [
            f"Keypoints in frame 0: {len(matched['keypoints0'])}",
            f"Keypoints in frame 1: {len(matched['keypoints1'])}",
            f"Accepted matches: {len(matched['matches'])}",
        ],
        "images": [_image("Accepted ORB matches", _png_bgr(matched["matched_view"]))],
        "notice": (
            "Each green line joins one point to its match. "
            "A lower ratio demands a clearer winner, so fewer matches remain."
        ),
    }


def run_torch(params: dict) -> dict:
    from physical_ai_lab.torch_bench import matmul_benchmark

    timed = matmul_benchmark(n=params["n"], seed=7)
    return {
        "call": f"matmul_benchmark(n={params['n']}, seed=7)",
        "lines": [
            f"Device: {timed['device']}",
            f"Input shape: {timed['input_shape']}",
            f"Output shape: {timed['output_shape']}",
            f"One matrix multiplication: {timed['elapsed_ms']:.2f} ms",
        ],
        "images": [],
        "notice": (
            "Both grids are square. The output has the same shape. "
            "A larger n multiplies more numbers, so the time grows quickly."
        ),
    }


def run_detection(params: dict) -> dict:
    from physical_ai_lab.detection import detect_objects

    frame = make_scene(640, 400)
    found = detect_objects(frame, threshold=float(params["threshold"]))
    scores = np.asarray(found["scores"], dtype=np.float64)
    highest = np.sort(scores)[::-1][:5] if scores.size else np.array([])
    score_text = ", ".join(f"{value:.3f}" for value in highest) if highest.size else "none"
    return {
        "call": f"detect_objects(frame, threshold={float(params['threshold']):.2f})",
        "lines": [
            f"Predictions at or above the threshold: {found['count']}",
            f"Highest scores, including those below the threshold: {score_text}",
        ],
        "images": [_image("Detector boxes", _png_bgr(found["visual_bgr"]))],
        "notice": (
            "The drawing contains boxes and a label, not a person or a car. "
            "Scores under the threshold are hidden. Lower the threshold to reveal weaker guesses."
        ),
    }


def run_camera(params: dict) -> dict:
    intrinsics = intrinsic_matrix(params["width"], params["height"], params["fx"], params["fy"])
    point = np.array([params["x_m"], params["y_m"], params["z_m"]], dtype=np.float64)
    pixel_u, pixel_v = project_point(point, intrinsics)
    recovered = back_project(pixel_u, pixel_v, float(point[2]), intrinsics)
    canvas = np.full((params["height"], params["width"], 3), 244, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (params["width"] - 1, params["height"] - 1), (30, 30, 30), 1)
    inside = 0 <= pixel_u < params["width"] and 0 <= pixel_v < params["height"]
    if inside:
        center = (int(round(pixel_u)), int(round(pixel_v)))
        cv2.drawMarker(canvas, center, (40, 70, 180), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)
        cv2.circle(canvas, center, 10, (40, 70, 180), 2)
    cv2.putText(
        canvas,
        f"pixel ({pixel_u:.0f}, {pixel_v:.0f})",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    place = "inside the image" if inside else "outside the image"
    return {
        "call": (
            f"K = intrinsic_matrix(width={params['width']}, height={params['height']}, "
            f"fx={params['fx']:.1f}, fy={params['fy']:.1f})\n"
            f"project_point([{params['x_m']:.2f}, {params['y_m']:.2f}, {params['z_m']:.2f}], K)\n"
            f"back_project(u, v, depth_m={params['z_m']:.2f}, K)"
        ),
        "lines": [
            f"3D point (m): [{params['x_m']:.2f}, {params['y_m']:.2f}, {params['z_m']:.2f}]",
            f"Pixel: ({pixel_u:.1f}, {pixel_v:.1f}), {place}",
            f"Back-projected point (m): [{recovered[0]:.4f}, {recovered[1]:.4f}, {recovered[2]:.4f}]",
        ],
        "images": [_image("Where the 3D point lands", _png_bgr(canvas))],
        "notice": (
            "Moving the point farther away, a larger Z, pulls the pixel toward the center. "
            "A larger focal length pushes it back toward the edge. Back-projection returns the same point."
        ),
    }


def run_depth(params: dict) -> dict:
    import mpl_toolkits.mplot3d  # noqa: F401

    depth = synthetic_depth(params["height"], params["width"])
    cx = params["width"] / 2
    cy = params["height"] / 2
    points = depth_to_points(depth, fx=params["fx"], fy=params["fy"], cx=cx, cy=cy)
    figure = plt.figure(figsize=(7.2, 5.2))
    axis = figure.add_subplot(111, projection="3d")
    sample = points[:: max(1, len(points) // 4000)]
    axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2, c=sample[:, 2], cmap="viridis")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title("Depth map turned into 3D points")
    figure.tight_layout()
    return {
        "call": (
            f"depth = synthetic_depth(height={params['height']}, width={params['width']})\n"
            f"depth_to_points(depth, fx={params['fx']:.1f}, fy={params['fy']:.1f}, "
            f"cx={cx:.1f}, cy={cy:.1f})"
        ),
        "lines": [
            f"Depth map shape: {depth.shape}",
            f"Point cloud shape: {points.shape}",
            f"Z range (m): {float(points[:, 2].min()):.3f} to {float(points[:, 2].max()):.3f}",
        ],
        "images": [_image("Point cloud", _png_figure(figure))],
        "notice": "Every pixel with a positive distance becomes one point. Color follows that distance, Z.",
    }


def run_motion(params: dict) -> dict:
    frame0 = make_scene(params["width"], params["height"])
    translated = np.roll(frame0, shift=(params["shift_y"], params["shift_x"]), axis=(0, 1))
    redrawn = make_scene(params["width"], params["height"], params["shift_x"], params["shift_y"])
    pure_dx, pure_dy, pure_response = estimate_shift(frame0, translated)
    scene_dx, scene_dy, scene_response = estimate_shift(frame0, redrawn)
    difference = cv2.absdiff(frame0, translated)
    return {
        "call": (
            f"translated = roll(frame0, shift_x={params['shift_x']}, shift_y={params['shift_y']})\n"
            "estimate_shift(frame0, translated)\n"
            "estimate_shift(frame0, redrawn_scene)"
        ),
        "lines": [
            (
                f"Pure slide of {params['shift_x']} px, {params['shift_y']} px: "
                f"dx={pure_dx:.2f}, dy={pure_dy:.2f}, response={pure_response:.3f}"
            ),
            (
                f"Redrawn scene: dx={scene_dx:.2f}, dy={scene_dy:.2f}, response={scene_response:.3f}"
            ),
        ],
        "images": [
            _image("Original", _png_bgr(frame0)),
            _image("Pure slide", _png_bgr(translated)),
            _image("Absolute difference of the pure slide", _png_bgr(difference)),
        ],
        "notice": (
            "On a pure slide, dx and dy follow the controls. "
            "The redrawn scene wraps shapes around the borders and leaves the label in place, so its estimate stays near zero."
        ),
    }


def run_quantize(params: dict) -> dict:
    weights = np.random.default_rng(7).normal(size=params["count"]).astype(np.float32)
    result = quantize_int8(weights, scale=float(params["scale"]))
    shown = min(80, weights.size)
    figure, axis = plt.subplots(figsize=(8.0, 3.2))
    axis.plot(weights[:shown], color="#1c1915", label="float32", linewidth=1.4)
    axis.plot(result["reconstructed"][:shown], color="#b4532a", label="reconstructed", linewidth=1.2)
    axis.set_title(f"First {shown} weights")
    axis.set_xlabel("index")
    axis.legend(frameon=False)
    figure.tight_layout()
    return {
        "call": f"quantize_int8(weights, scale={float(params['scale']):.1f})  # {params['count']:,} values",
        "lines": [
            f"Float32 storage: {result['fp32_bytes']:,} bytes",
            f"Int8 storage: {result['int8_bytes']:,} bytes",
            f"Storage ratio: {result['storage_ratio']:.1f}x",
            f"Mean absolute error: {result['mean_abs_error']:.6f}",
        ],
        "images": [_image("Original weights and the rounded copy", _png_figure(figure))],
        "notice": (
            "Int8 uses one byte per value and float32 uses four, so the ratio stays 4. "
            "A larger scale makes finer steps and a smaller error, until values fall outside -128 to 127 and get clipped."
        ),
    }


def run_pipeline(params: dict) -> dict:
    frame0 = make_scene(params["width"], params["height"])
    low = min(params["canny_low"], params["canny_high"])
    high = max(params["canny_low"], params["canny_high"])
    output = process_frame(
        frame0,
        nfeatures=params["nfeatures"],
        canny_low=low,
        canny_high=high,
    )
    visual = cv2.drawKeypoints(
        frame0,
        output["keypoints"],
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    frames = [
        make_scene(params["width"], params["height"], shift_x=index * 2, shift_y=index)
        for index in range(params["frames"])
    ]
    summary = benchmark_frames(frames, nfeatures=params["nfeatures"])
    lines = [
        f"Keypoints: {len(output['keypoints'])}",
        f"This frame, grayscale: {output['timings_ms']['grayscale_ms']:.2f} ms",
        f"This frame, edges: {output['timings_ms']['edges_ms']:.2f} ms",
        f"This frame, ORB: {output['timings_ms']['orb_ms']:.2f} ms",
    ]
    for stage, stats in summary["stages"].items():
        lines.append(f"{stage}: median {stats['median_ms']:.2f} ms, p95 {stats['p95_ms']:.2f} ms")
    lines.append(f"Median total: {summary['median_total_ms']:.2f} ms")
    lines.append(f"Approximate throughput: {summary['frames_per_second']:.1f} frames/s")
    return {
        "call": (
            f"process_frame(frame, nfeatures={params['nfeatures']}, "
            f"canny_low={low}, canny_high={high})\n"
            f"benchmark_frames(frames, nfeatures={params['nfeatures']})  # {params['frames']} frames"
        ),
        "lines": lines,
        "images": [
            _image("Edges", _png_bgr(output["edges"])),
            _image("ORB keypoints", _png_bgr(visual)),
        ],
        "notice": (
            "The median is the typical frame. The p95 is a slower frame, the time 95 percent of frames beat. "
            "Throughput uses the median total."
        ),
    }


def _param(
    name: str,
    label: str,
    kind: str,
    default: object,
    help_text: str,
    minimum: float | None = None,
    maximum: float | None = None,
    step: float | None = None,
) -> dict:
    item = {
        "name": name,
        "label": label,
        "kind": kind,
        "default": default,
        "help": help_text,
    }
    if minimum is not None:
        item["min"] = minimum
    if maximum is not None:
        item["max"] = maximum
    if step is not None:
        item["step"] = step
    return item


LESSONS: list[dict] = [
    {
        "id": "environment",
        "step": 0,
        "title": "Check the tools",
        "idea": (
            "This lesson asks the computer which versions of Python, NumPy, OpenCV, and PyTorch are installed. "
            "Every later lesson calls those tools. CUDA tells you whether a graphics chip is available for PyTorch."
        ),
        "watch": "Confirm the four libraries imported, then move to the next lesson.",
        "auto": True,
        "params": [],
        "run": run_environment,
    },
    {
        "id": "foundations",
        "step": 1,
        "title": "Numbers and units",
        "idea": (
            "Sensor readings arrive as lists of numbers. Normalize stretches a list so the smallest number becomes 0 "
            "and the largest becomes 1. The other two functions only change units."
        ),
        "watch": "Change one reading and see which bar hits 0 and which bar hits 1.",
        "auto": True,
        "params": [
            _param("readings", "Sensor readings", "numbers", "1.2, 1.4, 1.1, 1.8, 2.0", "Comma-separated measurements."),
            _param("distances_m", "Distances in meters", "numbers", "0.7, 1.4, 2.3, 4.0", "Each value is multiplied by 100."),
            _param("speed_mps", "Speed in m/s", "float", 1.0, "Multiplied by 3.6 to get km/h.", 0, 40, 0.1),
        ],
        "run": run_foundations,
    },
    {
        "id": "scene",
        "step": 2,
        "title": "A picture is an array",
        "idea": (
            "make_scene draws a grid, boxes, and a label so the vision lessons have a picture to study. "
            "The shape is height, width, and three color numbers per pixel."
        ),
        "watch": "Move the shift and compare the two drawings. Then read the crop shape.",
        "auto": True,
        "params": [
            _param("width", "Width in pixels", "int", 640, "Number of columns.", 160, 960, 16),
            _param("height", "Height in pixels", "int", 400, "Number of rows.", 120, 640, 8),
            _param("shift_x", "Shift right", "int", 8, "Pixels added to each drawn shape.", -80, 80, 1),
            _param("shift_y", "Shift down", "int", 3, "Pixels added to each drawn shape.", -40, 40, 1),
        ],
        "run": run_scene,
    },
    {
        "id": "profiling",
        "step": 3,
        "title": "Measure the time",
        "idea": (
            "The same sum of squares runs two ways. NumPy handles the whole array in one operation. "
            "The Python loop visits one number at a time."
        ),
        "watch": "The loop is given fewer numbers on purpose. Compare the times anyway.",
        "auto": True,
        "params": [
            _param("count", "Values for NumPy", "int", 1_000_000, "Length of the array passed to NumPy.", 10_000, 2_000_000, 10_000),
            _param("loop_count", "Values for the Python loop", "int", 100_000, "How many items the slow loop visits.", 1_000, 300_000, 1_000),
        ],
        "run": run_profiling,
    },
    {
        "id": "edges",
        "step": 4,
        "title": "Edges and lines",
        "idea": (
            "An edge is a place where brightness changes sharply. Canny marks those places. "
            "Hough then searches the marks for straight lines."
        ),
        "watch": "Lower the Canny thresholds to add marks. Raise the Hough threshold to drop lines.",
        "auto": True,
        "params": [
            _param("width", "Width", "int", 640, "Picture width.", 160, 960, 16),
            _param("height", "Height", "int", 400, "Picture height.", 120, 640, 8),
            _param("canny_low", "Canny low", "int", 60, "Fainter changes than this are ignored.", 0, 255, 1),
            _param("canny_high", "Canny high", "int", 150, "Strong changes above this are kept.", 0, 255, 1),
            _param("hough_threshold", "Hough threshold", "int", 45, "How many edge votes a line needs.", 5, 200, 1),
            _param("min_line_length", "Minimum line length", "int", 25, "Shorter segments are dropped.", 5, 200, 1),
            _param("max_line_gap", "Maximum gap", "int", 8, "A break smaller than this still counts as one line.", 0, 40, 1),
        ],
        "run": run_edges,
    },
    {
        "id": "matching",
        "step": 5,
        "title": "Match two pictures",
        "idea": (
            "ORB finds small distinctive points. The second picture is the same drawing, shifted. "
            "A match is kept when it is clearly better than the next-best candidate. The ratio is that test."
        ),
        "watch": "Lower the ratio and count how many green lines disappear.",
        "auto": True,
        "params": [
            _param("width", "Width", "int", 640, "Picture width.", 160, 960, 16),
            _param("height", "Height", "int", 400, "Picture height.", 120, 640, 8),
            _param("shift_x", "Shift right", "int", 8, "How far the second drawing moves.", -80, 80, 1),
            _param("shift_y", "Shift down", "int", 3, "How far the second drawing moves.", -40, 40, 1),
            _param("nfeatures", "ORB points requested", "int", 1000, "Upper limit on points per picture.", 50, 2000, 10),
            _param("ratio", "Match ratio", "float", 0.75, "Lower keeps only clearer matches.", 0.4, 0.95, 0.01),
        ],
        "run": run_matching,
    },
    {
        "id": "torch",
        "step": 6,
        "title": "PyTorch multiply",
        "idea": (
            "PyTorch stores grids of numbers, called tensors, and can multiply them. "
            "The size control sets both sides of two square grids."
        ),
        "watch": "Raise n and watch the time. The output shape stays square.",
        "auto": True,
        "params": [
            _param("n", "Matrix size n", "int", 512, "Both grids are n by n.", 32, 2048, 32),
        ],
        "run": run_torch,
    },
    {
        "id": "detection",
        "step": 7,
        "title": "Object detector",
        "idea": (
            "A pretrained network looks for objects such as a person, a car, or a chair. "
            "The lab picture is a drawing of boxes, so strong detections are rare. "
            "The first run loads the network and can take a few seconds."
        ),
        "watch": "Lower the threshold and see whether any weak score crosses it.",
        "auto": False,
        "params": [
            _param("threshold", "Confidence threshold", "float", 0.5, "Guesses below this are hidden.", 0.05, 0.95, 0.05),
        ],
        "run": run_detection,
    },
    {
        "id": "camera",
        "step": 8,
        "title": "Camera geometry",
        "idea": (
            "A pinhole camera turns a 3D point into a pixel. X is meters to the right, Y is meters down, "
            "and Z is meters in front of the camera. The focal lengths fx and fy set the zoom."
        ),
        "watch": "Increase Z and watch the mark move toward the center. Then increase fx.",
        "auto": True,
        "params": [
            _param("width", "Image width", "int", 640, "Used to place the image center.", 160, 960, 16),
            _param("height", "Image height", "int", 400, "Used to place the image center.", 120, 640, 8),
            _param("fx", "Focal length fx", "float", 520, "Horizontal zoom, in pixels.", 80, 1200, 10),
            _param("fy", "Focal length fy", "float", 520, "Vertical zoom, in pixels.", 80, 1200, 10),
            _param("x_m", "X meters", "float", 0.25, "Right of the camera center.", -2, 2, 0.05),
            _param("y_m", "Y meters", "float", -0.10, "Down from the camera center.", -2, 2, 0.05),
            _param("z_m", "Z meters", "float", 2.0, "Distance in front of the camera.", 0.2, 10, 0.1),
        ],
        "run": run_camera,
    },
    {
        "id": "depth",
        "step": 9,
        "title": "Depth to 3D",
        "idea": (
            "A depth map stores a distance for every pixel. With the focal length, each pixel becomes a 3D point. "
            "This distance field is synthetic: it grows gently toward the right and the bottom."
        ),
        "watch": "Change the focal length and see the cloud stretch sideways.",
        "auto": True,
        "params": [
            _param("width", "Depth width", "int", 240, "Columns in the depth map.", 40, 280, 8),
            _param("height", "Depth height", "int", 160, "Rows in the depth map.", 40, 200, 8),
            _param("fx", "Focal length fx", "float", 220, "Horizontal zoom used to back-project.", 40, 800, 10),
            _param("fy", "Focal length fy", "float", 220, "Vertical zoom used to back-project.", 40, 800, 10),
        ],
        "run": run_depth,
    },
    {
        "id": "motion",
        "step": 10,
        "title": "Image motion",
        "idea": (
            "Phase correlation estimates how far one picture slid relative to another. "
            "One comparison is a pure slide made by rolling the picture. The other comparison redraws the scene."
        ),
        "watch": "Set the slide to 15 and 0, then to 0 and 12. Read both estimates.",
        "auto": True,
        "params": [
            _param("width", "Width", "int", 640, "Picture width.", 160, 960, 16),
            _param("height", "Height", "int", 400, "Picture height.", 120, 640, 8),
            _param("shift_x", "Slide right", "int", 8, "Pixels of pure horizontal slide.", -40, 40, 1),
            _param("shift_y", "Slide down", "int", 3, "Pixels of pure vertical slide.", -30, 30, 1),
        ],
        "run": run_motion,
    },
    {
        "id": "quantize",
        "step": 11,
        "title": "Store smaller numbers",
        "idea": (
            "Float32 weights use four bytes each. Int8 quantization rounds them onto whole steps from -128 to 127 "
            "and stores one byte each. The scale chooses the step size."
        ),
        "watch": "Raise the scale and watch the mean error shrink, until clipping starts to fight it.",
        "auto": True,
        "params": [
            _param("count", "How many weights", "int", 100_000, "Random practice weights.", 1_000, 1_000_000, 1_000),
            _param("scale", "Scale", "float", 32, "Values are multiplied by this before rounding.", 1, 128, 1),
        ],
        "run": run_quantize,
    },
    {
        "id": "pipeline",
        "step": 12,
        "title": "One perception pipeline",
        "idea": (
            "The steps now run in order on one frame: grayscale, edges, then ORB points. "
            "The benchmark repeats that on several shifted frames and reports a typical time and a slower time."
        ),
        "watch": "Add frames or ORB points and see which stage takes the time.",
        "auto": True,
        "params": [
            _param("width", "Width", "int", 640, "Picture width.", 160, 960, 16),
            _param("height", "Height", "int", 400, "Picture height.", 120, 640, 8),
            _param("nfeatures", "ORB points", "int", 500, "Points requested in the pipeline.", 50, 2000, 10),
            _param("canny_low", "Canny low", "int", 60, "Lower edge threshold.", 0, 255, 1),
            _param("canny_high", "Canny high", "int", 150, "Upper edge threshold.", 0, 255, 1),
            _param("frames", "Benchmark frames", "int", 8, "How many shifted frames to time.", 1, 30, 1),
        ],
        "run": run_pipeline,
    },
]

BY_ID = {lesson["id"]: lesson for lesson in LESSONS}


def lesson_catalog() -> list[dict]:
    public = []
    for lesson in LESSONS:
        public.append(
            {
                "id": lesson["id"],
                "step": lesson["step"],
                "title": lesson["title"],
                "idea": lesson["idea"],
                "watch": lesson["watch"],
                "auto": lesson["auto"],
                "params": lesson["params"],
            }
        )
    return public


def _coerce(lesson: dict, raw: dict) -> dict:
    coerced: dict = {}
    for param in lesson["params"]:
        incoming = raw.get(param["name"], param["default"])
        if param["kind"] == "numbers":
            coerced[param["name"]] = str(incoming)
        elif param["kind"] == "int":
            coerced[param["name"]] = _clamp_number(incoming, param["min"], param["max"], as_int=True)
        elif param["kind"] == "float":
            coerced[param["name"]] = _clamp_number(incoming, param["min"], param["max"], as_int=False)
        else:
            raise ValueError(f"Unknown control {param['name']}")
    return coerced


def run_lesson(lesson_id: str, raw_params: dict | None = None) -> dict:
    lesson = BY_ID.get(lesson_id)
    if lesson is None:
        raise KeyError(lesson_id)
    params = _coerce(lesson, raw_params or {})
    started = time.perf_counter()
    result = lesson["run"](params)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
    result["params"] = params
    result["id"] = lesson_id
    return result
