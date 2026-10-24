"""Run the learning path from a terminal and write figures under results/."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from physical_ai_lab import __version__
from physical_ai_lab.camera import back_project, intrinsic_matrix, project_point
from physical_ai_lab.config import load_config, project_root
from physical_ai_lab.depth import depth_to_points, synthetic_depth
from physical_ai_lab.edges import detect_edges_and_lines
from physical_ai_lab.foundations import (
    DISTANCES_M,
    SENSOR_READINGS,
    meters_per_second_to_kmh,
    meters_to_centimeters,
    normalize,
)
from physical_ai_lab.matching import match_orb
from physical_ai_lab.motion import estimate_shift
from physical_ai_lab.pipeline import benchmark_frames, process_frame
from physical_ai_lab.profiling import time_square_sums
from physical_ai_lab.quantize import quantize_int8
from physical_ai_lab.scene import crop_and_resize, make_scene
from physical_ai_lab.torch_bench import matmul_benchmark


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return image[:, :, ::-1]


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Physical AI learning path.")
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run Faster R-CNN. The first call downloads pretrained weights.",
    )
    parser.add_argument("--frames", type=int, default=None, help="Frames in the benchmark.")
    parser.add_argument("--out", type=Path, default=None, help="Directory for figures and the report.")
    args = parser.parse_args(argv)

    config = load_config()
    root = project_root()
    out_dir = args.out or (root / "results")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_count = args.frames if args.frames is not None else config.benchmark_frames
    lines: list[str] = []

    def report(text: str = "") -> None:
        print(text)
        lines.append(text)

    import cv2
    import torch

    report(f"physical-ai-lab {__version__}")
    report(f"Python: {sys.version.split()[0]}")
    report(f"Platform: {platform.platform()}")
    report(f"NumPy: {np.__version__}")
    report(f"OpenCV: {cv2.__version__}")
    report(f"PyTorch: {torch.__version__}")
    report(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        report(f"GPU: {torch.cuda.get_device_name(0)}")

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    report("")
    report("1. Foundations")
    report(f"Normalized readings: {normalize(SENSOR_READINGS)}")
    report(f"Centimeters: {meters_to_centimeters(DISTANCES_M)}")
    report(f"1 m/s in km/h: {float(meters_per_second_to_kmh(np.array([1.0]))[0]):.1f}")

    frame0 = make_scene(config.width, config.height)
    frame1 = make_scene(config.width, config.height, config.scene_shift_x, config.scene_shift_y)
    views = crop_and_resize(frame0)
    report("")
    report("2. Images as arrays")
    report(f"Image shape: {frame0.shape} bytes: {frame0.nbytes}")
    report(f"Crop: {views['crop'].shape} bytes: {views['crop'].nbytes}")
    report(f"Half-size: {views['half'].shape} same shape as original: {views['half'].shape == frame0.shape}")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(_bgr_to_rgb(frame0))
    axes[0].set_title("Frame 0")
    axes[1].imshow(_bgr_to_rgb(frame1))
    axes[1].set_title("Frame 1: shifted scene")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(out_dir / "01_scene.png", dpi=120)
    plt.close(figure)

    rng = np.random.default_rng(config.seed)
    profile = time_square_sums(rng.random(1_000_000, dtype=np.float32))
    report("")
    report("3. Profiling")
    report(f"NumPy: {profile['numpy_ms']:.2f} ms for {int(profile['numpy_count']):,} values")
    report(f"Python loop: {profile['loop_ms']:.2f} ms for {int(profile['loop_count']):,} values")

    geometry = detect_edges_and_lines(
        frame0,
        canny_low=config.canny_low,
        canny_high=config.canny_high,
        hough_threshold=config.hough_threshold,
        min_line_length=config.hough_min_line_length,
        max_line_gap=config.hough_max_line_gap,
    )
    report("")
    report("4. Edges and lines")
    report(f"Detected lines: {geometry['line_count']}")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(geometry["gray"], cmap="gray")
    axes[0].set_title("Grayscale")
    axes[1].imshow(geometry["edges"], cmap="gray")
    axes[1].set_title("Canny edges")
    axes[2].imshow(_bgr_to_rgb(geometry["line_view"]))
    axes[2].set_title(f"Detected lines: {geometry['line_count']}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(out_dir / "02_edges.png", dpi=120)
    plt.close(figure)

    matched = match_orb(frame0, frame1, nfeatures=config.orb_features, ratio=config.match_ratio)
    report("")
    report("5. Feature matching")
    report(
        f"Keypoints: {len(matched['keypoints0'])} -> {len(matched['keypoints1'])} | "
        f"accepted matches: {len(matched['matches'])}"
    )
    figure, axis = plt.subplots(figsize=(15, 6))
    axis.imshow(_bgr_to_rgb(matched["matched_view"]))
    axis.set_title(f"ORB good matches: {len(matched['matches'])}")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(out_dir / "03_matches.png", dpi=120)
    plt.close(figure)

    bench = matmul_benchmark(n=512, seed=config.seed)
    report("")
    report("6. PyTorch")
    report(
        f"Device: {bench['device']} | {bench['input_shape']} -> {bench['output_shape']} | "
        f"{bench['elapsed_ms']:.2f} ms"
    )

    if args.detect:
        from physical_ai_lab.detection import detect_objects

        detected = detect_objects(frame0, threshold=config.detection_threshold)
        report("")
        report("7. Object detection")
        report(f"Predictions above threshold: {detected['count']}")
        cv2.imwrite(str(out_dir / "06_detections.png"), detected["visual_bgr"])

    intrinsics = intrinsic_matrix(config.width, config.height, config.fx, config.fy)
    point = np.array([0.25, -0.10, 2.0], dtype=np.float64)
    pixel_u, pixel_v = project_point(point, intrinsics)
    recovered = back_project(pixel_u, pixel_v, float(point[2]), intrinsics)
    report("")
    report("8. Camera geometry")
    report(f"3D point {point.tolist()} m -> pixel ({pixel_u:.1f}, {pixel_v:.1f})")
    report(f"Back-projected point (m): {recovered.tolist()}")

    depth = synthetic_depth()
    points = depth_to_points(depth, fx=220.0, fy=220.0, cx=depth.shape[1] / 2, cy=depth.shape[0] / 2)
    report("")
    report("9. Depth to 3D")
    report(f"Point cloud shape: {points.shape}")
    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    sample = points[::20]
    axis.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=2)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title("Synthetic depth back-projected to 3D")
    figure.tight_layout()
    figure.savefig(out_dir / "04_point_cloud.png", dpi=120)
    plt.close(figure)

    translated = np.roll(frame0, shift=(config.scene_shift_y, config.scene_shift_x), axis=(0, 1))
    dx, dy, response = estimate_shift(frame0, translated)
    scene_dx, scene_dy, scene_response = estimate_shift(frame0, frame1)
    report("")
    report("10. Motion")
    report(
        f"Pure translation of +{config.scene_shift_x} px, +{config.scene_shift_y} px: "
        f"dx={dx:.2f}px, dy={dy:.2f}px, response={response:.3f}"
    )
    report(
        f"Redrawn scene (shapes wrap, label stays fixed): "
        f"dx={scene_dx:.2f}px, dy={scene_dy:.2f}px, response={scene_response:.3f}"
    )

    weights = np.random.default_rng(config.seed).normal(size=1_000_000).astype(np.float32)
    quantized = quantize_int8(weights, scale=config.quant_scale)
    report("")
    report("11. Quantization")
    report(f"FP32 storage: {quantized['fp32_bytes']} bytes")
    report(f"INT8 storage: {quantized['int8_bytes']} bytes")
    report(f"Storage ratio: {quantized['storage_ratio']:.1f}x")
    report(f"Mean absolute quantization error: {quantized['mean_abs_error']}")

    output = process_frame(frame0, nfeatures=config.pipeline_orb_features)
    report("")
    report("12. Integrated pipeline")
    report(f"Keypoints: {len(output['keypoints'])}")
    report(f"Stage timings: {output['timings_ms']}")
    visual = cv2.drawKeypoints(
        frame0,
        output["keypoints"],
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(output["edges"], cmap="gray")
    axes[0].set_title("Edges")
    axes[1].imshow(_bgr_to_rgb(visual))
    axes[1].set_title("ORB keypoints")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(out_dir / "05_pipeline.png", dpi=120)
    plt.close(figure)

    frames = [
        make_scene(config.width, config.height, shift_x=i * 2, shift_y=i) for i in range(frame_count)
    ]
    summary = benchmark_frames(frames, nfeatures=config.pipeline_orb_features)
    report("")
    report("13. Benchmark")
    for stage, stats in summary["stages"].items():
        report(f"{stage:16s} median={stats['median_ms']:7.2f} ms | p95={stats['p95_ms']:7.2f} ms")
    report(f"Median stage total: {summary['median_total_ms']:.2f} ms")
    report(f"Approximate throughput: {summary['frames_per_second']:.1f} frames/s")
    report("")
    report(f"Figures written to {out_dir}")

    (out_dir / "lab_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
