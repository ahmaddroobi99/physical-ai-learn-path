import numpy as np

from physical_ai_lab.pipeline import benchmark_frames, process_frame
from physical_ai_lab.profiling import time_square_sums
from physical_ai_lab.quantize import quantize_int8
from physical_ai_lab.scene import make_scene
from physical_ai_lab.torch_bench import matmul_benchmark


def test_square_sum_numpy_matches_a_short_loop():
    values = np.arange(20, dtype=np.float32)
    timed = time_square_sums(values, loop_count=20)
    assert np.isclose(timed["numpy_result"], timed["loop_result"])


def test_int8_storage_is_four_times_smaller():
    weights = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    result = quantize_int8(weights, scale=32.0)
    assert result["storage_ratio"] == 4.0
    assert result["quantized"].dtype == np.int8
    assert result["mean_abs_error"] < 1.0 / 32.0


def test_pipeline_and_benchmark():
    frames = [make_scene(shift_x=i, shift_y=0) for i in range(3)]
    output = process_frame(frames[0], nfeatures=100)
    assert set(output["timings_ms"]) == {"grayscale_ms", "edges_ms", "orb_ms"}
    assert len(output["keypoints"]) > 0
    summary = benchmark_frames(frames, nfeatures=100)
    assert summary["median_total_ms"] > 0
    assert summary["frames_per_second"] > 0


def test_small_matmul_reports_shape():
    result = matmul_benchmark(n=32, seed=7)
    assert result["output_shape"] == (32, 32)
    assert result["elapsed_ms"] >= 0
